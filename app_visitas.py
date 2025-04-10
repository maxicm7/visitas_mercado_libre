# app.py
import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
from io import StringIO # Para CSV en memoria
import datetime
import hashlib
import base64
import os
import urllib.parse
import sys # Para depuración opcional

# --- Configuración ---
st.set_page_config(layout="wide")
st.title("Analizador de Visitas de Competencia en Mercado Libre")

# Acceder a los secretos usando st.secrets de forma segura
try:
    CLIENT_ID = st.secrets["meli_client_id"]
    CLIENT_SECRET = st.secrets["meli_client_secret"]
except KeyError as e:
    st.error(f"Error Crítico: Falta el secreto '{e.args[0]}' en la configuración.")
    st.error("Por favor, asegúrate de haber configurado 'meli_client_id' y 'meli_client_secret' en Streamlit Cloud (Settings -> Secrets) o en tu archivo .streamlit/secrets.toml local.")
    st.stop()
except Exception as e:
    st.error(f"Error Crítico: Ocurrió un error inesperado al leer los secretos: {e}")
    st.stop()

# Otras configuraciones
REDIRECT_URI = 'https://lorenzoautomotores.com.ar/'
TOKEN_URL = 'https://api.mercadolibre.com/oauth/token'
AUTH_BASE_URL = "https://auth.mercadolibre.com.ar/authorization?"
VISITS_URL_TEMPLATE = 'https://api.mercadolibre.com/items/{item_id}/visits/time_window'
API_CALL_DELAY = 0.4

# --- Funciones de Autenticación ---
def generate_code_challenge():
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b'=')
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier).digest()
    ).rstrip(b'=')
    return code_verifier.decode(), code_challenge.decode()

def get_authorization_url(client_id, redirect_uri, code_challenge):
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256"
    }
    return AUTH_BASE_URL + urllib.parse.urlencode(params)

def obtener_token_de_acceso(authorization_code, code_verifier, client_id, client_secret, redirect_uri):
    params = {
        'grant_type': 'authorization_code',
        'client_id': client_id,
        'client_secret': client_secret,
        'code': authorization_code,
        'redirect_uri': redirect_uri,
        'code_verifier': code_verifier,
    }
    try:
        response = requests.post(TOKEN_URL, data=params, timeout=15)
        response.raise_for_status()
        token_data = response.json()
        expires_in = token_data.get('expires_in', 3600)
        st.session_state.token_obtained_at = time.time()
        st.session_state.token_expires_in = expires_in
        return token_data.get('access_token'), token_data.get('refresh_token'), token_data
    except requests.exceptions.RequestException as e:
        error_detail = "No response"
        if e.response is not None:
            try: error_detail = e.response.json()
            except ValueError: error_detail = e.response.text
        st.error(f"Error de red/HTTP al obtener token: {e}")
        st.error(f"Detalles: {error_detail}")
        return None, None, None
    except Exception as e:
        st.error(f"Error inesperado al procesar token: {e}")
        return None, None, None

# --- Función de API de Visitas (Llamadas individuales) ---
# @st.cache_data(ttl=3600)
def fetch_visits(_access_token, item_ids, date_from_str, date_to_str):
    if not _access_token:
        st.error("Error: Access Token no disponible para buscar visitas.")
        return {}
    if not item_ids:
        st.warning("No se proporcionaron IDs para buscar visitas.")
        return {}

    headers = {'Authorization': f'Bearer {_access_token}'}
    visits_dict = {}
    item_ids_list = list(set(item_ids))

    try:
        start_date = datetime.datetime.strptime(date_from_str, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(date_to_str, '%Y-%m-%d').date()
        delta = end_date - start_date
        days_diff = delta.days + 1
    except ValueError:
        st.error("Formato de fecha inválido.")
        return {}

    params_base = {'unit': 'day', 'ending': date_to_str}
    if days_diff <= 60 and days_diff > 0:
        params_base['last'] = str(days_diff)
    elif days_diff > 0 :
        st.warning(f"Rango ({days_diff} días) > 60. Usando últimos 60 días hasta {date_to_str}.")
        params_base['last'] = '60'
    else:
        st.error("Rango de fechas inválido (debe ser al menos 1 día).")
        return {}

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_ids = len(item_ids_list)

    for index, item_id in enumerate(item_ids_list):
        if not item_id or pd.isna(item_id):
             # st.warning(f"Saltando ID inválido/vacío en índice {index}.") # Comentado para no llenar de warnings
             continue

        single_item_url = VISITS_URL_TEMPLATE.format(item_id=item_id)
        params = params_base.copy()

        try:
            status_text.text(f"Procesando ID {index + 1}/{total_ids}: {item_id}...")
            response = requests.get(single_item_url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            total_visits = 0
            if isinstance(data, dict):
                 total_visits_val = data.get('total_visits')
                 if total_visits_val is not None:
                     total_visits = total_visits_val
                 elif 'results' in data and isinstance(data['results'], list):
                      total_visits = sum(day_data.get('total', 0) for day_data in data['results'])
                 else:
                      total_visits = data.get('visits', 0)
            visits_dict[item_id] = int(total_visits) if pd.notna(total_visits) else 0

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                visits_dict[item_id] = 0
            elif e.response.status_code == 429 or e.response.status_code == 403:
                 st.error(f"Error Rate Limit/Prohibido ({e.response.status_code}) para ID {item_id}. Intentando más despacio...")
                 time.sleep(5)
                 visits_dict[item_id] = -2
            else:
                error_msg = f"Error HTTP {e.response.status_code}"
                try: error_detail = e.response.json(); error_msg += f": {error_detail.get('message', e.response.text)}"
                except ValueError: error_msg += f": {e.response.text}"
                st.error(f"{error_msg} (ID: {item_id})")
                visits_dict[item_id] = -1
        except requests.exceptions.RequestException as e:
            st.error(f"Error de red/timeout para ID {item_id}: {e}")
            visits_dict[item_id] = -1
        except Exception as e:
             st.error(f"Error inesperado procesando ID {item_id}: {e}")
             visits_dict[item_id] = -1

        progress = (index + 1) / total_ids
        progress_bar.progress(progress)
        time.sleep(API_CALL_DELAY)

    progress_bar.empty()
    status_text.text(f"Proceso completado. {len(visits_dict)} IDs procesados.")
    return visits_dict

# --- Función Auxiliar para CSV ---
def dataframe_to_csv_bytes(df):
    output = StringIO()
    df.to_csv(output, index=False, encoding='utf-8')
    csv_data = output.getvalue().encode('utf-8')
    output.close()
    return csv_data

# --- Inicialización de Session State ---
if 'access_token' not in st.session_state: st.session_state.access_token = None
if 'refresh_token' not in st.session_state: st.session_state.refresh_token = None
if 'code_verifier' not in st.session_state: st.session_state.code_verifier = None
if 'authorization_url' not in st.session_state: st.session_state.authorization_url = None
if 'authentication_step' not in st.session_state: st.session_state.authentication_step = 1
if 'uploaded_data' not in st.session_state: st.session_state.uploaded_data = None
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'token_obtained_at' not in st.session_state: st.session_state.token_obtained_at = None
if 'token_expires_in' not in st.session_state: st.session_state.token_expires_in = None
if 'current_uploaded_filename' not in st.session_state: st.session_state.current_uploaded_filename = None


# --- Interfaz de Streamlit ---

# --- Sección 1: Autenticación ---
with st.expander("1. Autenticación con Mercado Libre", expanded=not st.session_state.access_token):

    if st.session_state.access_token:
        st.success(f"Autenticado correctamente.")
        if st.session_state.token_obtained_at and st.session_state.token_expires_in:
            remaining_time = int(st.session_state.token_expires_in - (time.time() - st.session_state.token_obtained_at))
            if remaining_time > 0: st.info(f"Token válido por ~{remaining_time // 60} min.")
            else: st.warning("El token puede haber expirado. La próxima acción podría requerir re-autenticar.")
        st.session_state.authentication_step = 3
        if st.button("Cerrar Sesión / Re-Autenticar"):
             for key in list(st.session_state.keys()):
                 if key in ['access_token', 'refresh_token', 'code_verifier', 'authorization_url',
                           'token_obtained_at', 'token_expires_in', 'authentication_step',
                           'uploaded_data', 'results_df', 'current_uploaded_filename']:
                     del st.session_state[key]
             st.rerun()

    else:
        if st.session_state.authentication_step == 1:
            st.info("Haz clic en el botón para iniciar el proceso de autenticación con Mercado Libre.")
            if st.button("Iniciar Autenticación"):
                st.session_state.code_verifier, code_challenge = generate_code_challenge()
                st.session_state.authorization_url = get_authorization_url(CLIENT_ID, REDIRECT_URI, code_challenge)
                st.session_state.authentication_step = 2
                st.rerun()

        if st.session_state.authentication_step == 2:
            st.markdown("#### Pasos para Autenticar:")
            st.markdown("1. Haz clic en el enlace de abajo para ir a Mercado Libre.")
            st.markdown(f"<a href='{st.session_state.authorization_url}' target='_blank'>Abrir URL de Autorización Mercado Libre</a>", unsafe_allow_html=True)
            st.markdown("2. Inicia sesión en Mercado Libre (si es necesario) y **autoriza** la aplicación.")
            st.markdown(f"3. Serás redirigido a `{REDIRECT_URI}`. **Copia** el código largo que aparece después de `?code=` en la barra de direcciones de esa página.")
            st.markdown("4. **Pega** ese código en el campo de abajo y haz clic en 'Obtener Token'.")
            st.warning("Asegúrate de copiar **solo** el valor del parámetro 'code'.")
            authorization_code = st.text_input("Pega el código ('code') aquí:", key="auth_code_input", type="password")

            if st.button("Obtener Access Token", key="get_token_button"):
                if authorization_code and st.session_state.code_verifier:
                    with st.spinner("Verificando código y obteniendo token de acceso..."):
                        access_token, refresh_token, _ = obtener_token_de_acceso(
                            authorization_code, st.session_state.code_verifier,
                            CLIENT_ID, CLIENT_SECRET, REDIRECT_URI
                        )
                    if access_token:
                        st.session_state.access_token = access_token
                        st.session_state.refresh_token = refresh_token
                        st.session_state.authentication_step = 3
                        st.session_state.code_verifier = None
                        st.session_state.authorization_url = None
                        st.success("¡Autenticación Exitosa!")
                        time.sleep(1.5)
                        st.rerun()
                else:
                    st.warning("Pega el código de autorización obtenido antes de presionar el botón.")

# --- Sección 2: Carga de Archivo (Solo si está autenticado) ---
if st.session_state.get('authentication_step') == 3:
    st.header("2. Cargar Archivo Excel")
    uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx) con columnas 'id', 'OEM', 'Seller2', etc.", type=["xlsx"], key="file_uploader")

    if uploaded_file:
        if st.session_state.current_uploaded_filename != uploaded_file.name:
             st.session_state.results_df = None
             st.session_state.uploaded_data = None
             st.session_state.current_uploaded_filename = uploaded_file.name

        if st.session_state.uploaded_data is None:
            try:
                with st.spinner(f"Leyendo archivo '{uploaded_file.name}'..."):
                    df = pd.read_excel(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.success(f"Archivo '{uploaded_file.name}' cargado.")

                    # --- Validación y Preparación de Datos ---
                    # Define las columnas requeridas EN MINÚSCULAS
                    required_columns = ['id', 'oem', 'seller2'] # <-- CORRECCIÓN APLICADA

                    # Normalizar nombres de columna del archivo a minúsculas
                    st.session_state.uploaded_data.columns = [col.strip().lower() for col in st.session_state.uploaded_data.columns]
                    current_columns = st.session_state.uploaded_data.columns

                    # Verificar si las columnas requeridas (en minúsculas) están presentes
                    missing_cols = [col for col in required_columns if col not in current_columns]

                    if missing_cols:
                        # Mostrar qué columnas requeridas (en minúsculas) faltan
                        st.error(f"El archivo Excel debe contener las columnas (ignorando mayús/minús): {', '.join(required_columns)}. Faltan o tienen nombres diferentes: {', '.join(missing_cols)}")
                        st.session_state.uploaded_data = None
                    else:
                        try:
                            # Asegurar que 'id' es string y no está vacío
                            st.session_state.uploaded_data['id'] = st.session_state.uploaded_data['id'].astype(str).str.strip()
                            if st.session_state.uploaded_data['id'].eq('').any():
                                st.warning("Se encontraron filas con 'id' vacío. Estas filas no serán procesadas.")
                        except Exception as e:
                             st.error(f"Error al procesar la columna 'id': {e}")
                             st.session_state.uploaded_data = None

            except Exception as e:
                st.error(f"Error al leer el archivo Excel: {e}")
                st.session_state.uploaded_data = None

        if st.session_state.uploaded_data is not None:
             st.dataframe(st.session_state.uploaded_data.head())


# --- Sección 3: Filtrar y Obtener Visitas (Solo si hay datos cargados y válidos) ---
df_original = st.session_state.get('uploaded_data')

if df_original is not None:
    st.header("3. Filtrar Datos y Obtener Visitas")

    # 3.1 Seleccionar OEM (usando columna 'oem' normalizada)
    try:
        oem_column = df_original['oem'].fillna('Desconocido').astype(str)
        available_oems = sorted(oem_column.unique())
        selected_oem = st.selectbox("Selecciona el OEM:", options=available_oems, key="oem_selector")
    except KeyError:
        st.error("Columna 'oem' no encontrada. Revisa el archivo subido.")
        st.stop()

    # 3.2 Seleccionar Fechas
    today = datetime.date.today()
    if today.day < 5 :
         last_day_prev_month = today.replace(day=1) - datetime.timedelta(days=1)
         default_start_date = last_day_prev_month.replace(day=1)
         default_end_date = last_day_prev_month
    else:
         default_start_date = today.replace(day=1)
         default_end_date = today - datetime.timedelta(days=1) if today.day > 1 else today.replace(day=1)

    col1, col2 = st.columns(2)
    with col1: date_from = st.date_input("Fecha Desde:", value=default_start_date, max_value=today, key="date_from")
    with col2: date_to = st.date_input("Fecha Hasta:", value=default_end_date, min_value=date_from, max_value=today, key="date_to")

    if date_to < date_from:
        st.error("La 'Fecha Hasta' no puede ser anterior a la 'Fecha Desde'.")
        st.stop()

    date_from_str = date_from.strftime('%Y-%m-%d')
    date_to_str = date_to.strftime('%Y-%m-%d')

    # 3.3 Botón para iniciar búsqueda
    st.markdown("---")
    if st.button(f"Buscar Visitas para OEM '{selected_oem}' ({date_from_str} a {date_to_str})", key="fetch_visits_button"):
        st.session_state.results_df = None

        # Filtrar usando columna 'oem' normalizada
        filtered_df = df_original[df_original['oem'].astype(str) == str(selected_oem)].copy()
        st.info(f"Filtrado por OEM '{selected_oem}'. {len(filtered_df)} filas encontradas.")

        if filtered_df.empty:
            st.warning(f"No se encontraron registros para el OEM '{selected_oem}'.")
            st.session_state.results_df = pd.DataFrame()
        else:
            if 'id' in filtered_df.columns:
                 # Obtener IDs únicos, asegurándose de que sean strings y no vacíos
                 item_ids_to_fetch = filtered_df['id'].astype(str).str.strip().dropna().unique().tolist()
                 item_ids_to_fetch = [item_id for item_id in item_ids_to_fetch if item_id] # Filtrar strings vacíos

                 if not item_ids_to_fetch:
                      st.warning("No hay IDs válidos en los datos filtrados para buscar.")
                      st.session_state.results_df = pd.DataFrame()
                 else:
                     with st.spinner(f"Consultando API Mercado Libre para {len(item_ids_to_fetch)} IDs (puede tardar)..."):
                          visits_data = fetch_visits(
                              st.session_state.access_token, item_ids_to_fetch,
                              date_from_str, date_to_str
                          )

                     visits_series = pd.Series(visits_data, name='visits')
                     visits_series.index.name = 'id'
                     visits_series.index = visits_series.index.astype(str)

                     filtered_df['id'] = filtered_df['id'].astype(str)
                     results_df_merged = pd.merge(filtered_df, visits_series, on='id', how='left')
                     results_df_merged['visits'] = pd.to_numeric(results_df_merged['visits'], errors='coerce').fillna(0).astype('Int64')

                     st.session_state.results_df = results_df_merged
                     st.success("¡Búsqueda de visitas completada!")
            else:
                 st.error("Columna 'id' no encontrada después de filtrar.")
                 st.session_state.results_df = None

# --- Sección 4: Resultados y Visualización ---
results_df = st.session_state.get('results_df')

if results_df is not None:
    st.header("4. Resultados")

    if results_df.empty:
        st.info("No se generaron resultados.")
    else:
        results_df_display = results_df.copy()
        if 'seller2' in results_df_display.columns:
             results_df_display = results_df_display.rename(columns={'seller2': 'Nombre Competidor'})
             results_df_display['Nombre Competidor'] = results_df_display['Nombre Competidor'].fillna('Desconocido')
        else:
             results_df_display['Nombre Competidor'] = 'Desconocido'

        results_df_display['visits'] = pd.to_numeric(results_df_display['visits'], errors='coerce').fillna(0)

        if 'Nombre Competidor' in results_df_display.columns:
             visits_by_competitor = results_df_display.groupby('Nombre Competidor')['visits'].sum().reset_index()
             visits_by_competitor = visits_by_competitor.sort_values(by='visits', ascending=False)

             st.subheader("Gráfico de Visitas por Competidor")
             if not visits_by_competitor.empty and visits_by_competitor['visits'].sum() > 0:
                  fig = px.bar(visits_by_competitor.head(30), x='Nombre Competidor', y='visits',
                               title=f"Visitas por Competidor (Top 30) - OEM: {selected_oem} ({date_from_str} a {date_to_str})",
                               labels={'visits': 'Total Visitas'})
                  fig.update_layout(xaxis_tickangle=-45)
                  st.plotly_chart(fig, use_container_width=True)
             else:
                  st.info("No hay datos de visitas para graficar.")
        else:
              st.warning("No se pudo generar gráfico (falta columna 'Nombre Competidor').")

        st.subheader("Tabla Completa con Visitas")
        display_cols = ['id', 'title', 'seller_id', 'category_id', 'Nombre Competidor', 'oem', 'description', 'link', 'visits']
        existing_display_cols = [col for col in display_cols if col in results_df_display.columns]
        st.dataframe(results_df_display[existing_display_cols])

        try:
            csv_bytes = dataframe_to_csv_bytes(results_df_display[existing_display_cols])
            st.download_button(
                label="Descargar Resultados en CSV",
                data=csv_bytes,
                file_name=f"resultados_visitas_{selected_oem.replace(' ','_')}_{date_from_str}_a_{date_to_str}.csv",
                mime="text/csv"
            )
        except Exception as e:
             st.error(f"Error al generar CSV: {e}")

# --- Pie de Página ---
st.markdown("---")
st.caption("Aplicación de Análisis de Visitas v1.0")
