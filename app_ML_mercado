# app_candidatos.py
"""
Analizador de Productos Candidatos - Mercado Libre
Combina:
  1. Demanda / Competencia (via /sites/{SITE}/search)
  2. Tendencias de búsqueda (via /trends/{SITE}/{category_id})
  3. Calculadora de rentabilidad real (via /sites/{SITE}/listing_prices)
Y arma un ranking de "mejores candidatos" para vender.
"""
import streamlit as st
import pandas as pd
import requests
import time
import datetime
import hashlib
import base64
import os
import urllib.parse

st.set_page_config(layout="wide")
st.title("🔎 Buscador de Productos Candidatos - Mercado Libre")

# ==============================================================================
# --- CONFIGURACIÓN ---
# ==============================================================================
try:
    CLIENT_ID = st.secrets["meli_client_id"]
    CLIENT_SECRET = st.secrets["meli_client_secret"]
except KeyError as e:
    st.error(f"Error Crítico: Falta el secreto '{e.args[0]}' en la configuración.")
    st.error("Configurá 'meli_client_id' y 'meli_client_secret' en Streamlit Cloud (Settings -> Secrets).")
    st.stop()

REDIRECT_URI = 'https://lorenzoautomotores.com.ar/'
TOKEN_URL = 'https://api.mercadolibre.com/oauth/token'
AUTH_BASE_URL = "https://auth.mercadolibre.com.ar/authorization?"
SITE_ID = "MLA"  # Cambiar si operás en otro país (MLB, MLM, MCO, etc.)
SEARCH_URL_TEMPLATE = f'https://api.mercadolibre.com/sites/{SITE_ID}/search'
TRENDS_URL_TEMPLATE = f'https://api.mercadolibre.com/trends/{SITE_ID}/{{category_id}}'
LISTING_PRICES_URL = f'https://api.mercadolibre.com/sites/{SITE_ID}/listing_prices'
CATEGORIES_URL = f'https://api.mercadolibre.com/sites/{SITE_ID}/categories'
API_CALL_DELAY = 0.4

# ==============================================================================
# --- AUTENTICACIÓN (idéntica a la app original, para reusar sesión/token) ---
# ==============================================================================
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
        return token_data.get('access_token'), token_data.get('refresh_token')
    except requests.exceptions.RequestException as e:
        error_detail = "No response"
        if e.response is not None:
            try: error_detail = e.response.json()
            except ValueError: error_detail = e.response.text
        st.error(f"Error de red/HTTP al obtener token: {e}")
        st.error(f"Detalles: {error_detail}")
        return None, None
    except Exception as e:
        st.error(f"Error inesperado al procesar token: {e}")
        return None, None

# --- Estado de sesión ---
for key, default in [
    ('access_token', None), ('refresh_token', None), ('code_verifier', None),
    ('authorization_url', None), ('authentication_step', 1),
    ('token_obtained_at', None), ('token_expires_in', None),
    ('candidatos_df', None), ('categorias_cache', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.expander("1. Autenticación con Mercado Libre", expanded=not st.session_state.access_token):
    if st.session_state.access_token:
        st.success("Autenticado correctamente.")
        if st.session_state.token_obtained_at and st.session_state.token_expires_in:
            remaining = int(st.session_state.token_expires_in - (time.time() - st.session_state.token_obtained_at))
            if remaining > 0:
                st.info(f"Token válido por ~{remaining // 60} min.")
            else:
                st.warning("El token puede haber expirado. Re-autenticá si falla alguna consulta.")
        st.session_state.authentication_step = 3
        if st.button("Cerrar Sesión / Re-Autenticar"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    else:
        if st.session_state.authentication_step == 1:
            st.info("Hacé clic para iniciar el proceso de autenticación con Mercado Libre.")
            if st.button("Iniciar Autenticación"):
                st.session_state.code_verifier, code_challenge = generate_code_challenge()
                st.session_state.authorization_url = get_authorization_url(CLIENT_ID, REDIRECT_URI, code_challenge)
                st.session_state.authentication_step = 2
                st.rerun()
        if st.session_state.authentication_step == 2:
            st.markdown("#### Pasos:")
            st.markdown(f"<a href='{st.session_state.authorization_url}' target='_blank'>Abrir URL de Autorización</a>", unsafe_allow_html=True)
            st.markdown(f"Copiá el código (`?code=...`) que aparece al ser redirigido a `{REDIRECT_URI}`.")
            authorization_code = st.text_input("Pegá el código aquí:", key="auth_code_input", type="password")
            if st.button("Obtener Access Token"):
                if authorization_code and st.session_state.code_verifier:
                    with st.spinner("Obteniendo token..."):
                        access_token, refresh_token = obtener_token_de_acceso(
                            authorization_code, st.session_state.code_verifier,
                            CLIENT_ID, CLIENT_SECRET, REDIRECT_URI
                        )
                    if access_token:
                        st.session_state.access_token = access_token
                        st.session_state.refresh_token = refresh_token
                        st.session_state.authentication_step = 3
                        st.success("¡Autenticación Exitosa!")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("Pegá el código de autorización antes de continuar.")

if st.session_state.authentication_step != 3:
    st.stop()

headers = {'Authorization': f"Bearer {st.session_state.access_token}"}

# ==============================================================================
# --- FUNCIONES DE ANÁLISIS ---
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def buscar_categorias(_headers):
    try:
        r = requests.get(CATEGORIES_URL, headers=_headers, timeout=15)
        r.raise_for_status()
        return {c['name']: c['id'] for c in r.json()}
    except Exception as e:
        st.warning(f"No se pudieron cargar categorías: {e}")
        return {}

def analizar_keyword(keyword, category_id, _headers):
    """Devuelve métricas de demanda/competencia para una keyword."""
    params = {'q': keyword, 'limit': 50}
    if category_id:
        params['category'] = category_id
    try:
        r = requests.get(SEARCH_URL_TEMPLATE, headers=_headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = data.get('results', [])
        total_publicaciones = data.get('paging', {}).get('total', len(results))
        precios = [it.get('price', 0) for it in results if it.get('price')]
        vendidos = [it.get('sold_quantity', 0) for it in results]
        return {
            'keyword': keyword,
            'competencia_publicaciones': total_publicaciones,
            'precio_promedio': round(sum(precios) / len(precios), 2) if precios else 0,
            'precio_min': min(precios) if precios else 0,
            'precio_max': max(precios) if precios else 0,
            'demanda_vendidos_top50': sum(vendidos),
            'top_item_id': results[0]['id'] if results else None,
            'top_category_id': results[0].get('category_id') if results else category_id,
            'error': None,
        }
    except requests.exceptions.RequestException as e:
        return {'keyword': keyword, 'error': str(e)}

def obtener_tendencias(category_id, _headers):
    """Trae el top 50 de tendencias de una categoría."""
    if not category_id:
        return []
    try:
        url = TRENDS_URL_TEMPLATE.format(category_id=category_id)
        r = requests.get(url, headers=_headers, timeout=15)
        r.raise_for_status()
        return [t.get('keyword', '').lower() for t in r.json()]
    except requests.exceptions.RequestException:
        return []

def calcular_rentabilidad(precio_venta, category_id, costo_producto, peso_gramos,
                           listing_type_id, logistic_type, shipping_mode, _headers):
    """Llama a /listing_prices (la calculadora oficial) y devuelve el desglose."""
    params = {
        'price': precio_venta,
        'category_id': category_id,
        'listing_type_id': listing_type_id,
        'logistic_type': logistic_type,
        'shipping_mode': shipping_mode,
        'billable_weight': peso_gramos,
    }
    try:
        r = requests.get(LISTING_PRICES_URL, headers=_headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        # La API puede devolver una lista de listing_types disponibles
        registro = data[0] if isinstance(data, list) else data
        comision = registro.get('sale_fee', 0)
        costo_envio = registro.get('shipping_fee', registro.get('shipping', {}).get('cost', 0) if isinstance(registro.get('shipping'), dict) else 0)
        margen_neto = precio_venta - comision - (costo_envio or 0) - costo_producto
        margen_pct = (margen_neto / precio_venta * 100) if precio_venta else 0
        return {
            'comision_meli': comision, 'costo_envio_estimado': costo_envio or 0,
            'margen_neto': round(margen_neto, 2), 'margen_pct': round(margen_pct, 1),
            'error': None, 'raw': registro,
        }
    except requests.exceptions.RequestException as e:
        detail = ""
        if e.response is not None:
            try: detail = e.response.json()
            except ValueError: detail = e.response.text
        return {'error': f"{e} | {detail}"}

# ==============================================================================
# --- INTERFAZ: SECCIÓN 2 - BÚSQUEDA DE CANDIDATOS ---
# ==============================================================================
st.header("2. Analizar Palabras Clave / Productos Candidatos")

categorias = buscar_categorias(headers)
col_a, col_b = st.columns([2, 1])
with col_a:
    keywords_input = st.text_area(
        "Palabras clave a evaluar (una por línea):",
        placeholder="ventilador de techo silencioso\nfunda notebook 15 pulgadas\ncargador inalámbrico auto",
        height=120,
    )
with col_b:
    categoria_nombre = st.selectbox("Categoría (opcional, mejora precisión):",
                                     options=["(Todas)"] + sorted(categorias.keys()) if categorias else ["(Todas)"])
    category_id = categorias.get(categoria_nombre) if categoria_nombre != "(Todas)" else None

if st.button("🔍 Analizar Demanda y Competencia"):
    keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]
    if not keywords:
        st.warning("Ingresá al menos una palabra clave.")
    else:
        resultados = []
        progress = st.progress(0)
        for i, kw in enumerate(keywords):
            res = analizar_keyword(kw, category_id, headers)
            resultados.append(res)
            progress.progress((i + 1) / len(keywords))
            time.sleep(API_CALL_DELAY)
        progress.empty()

        df = pd.DataFrame(resultados)
        errores = df[df['error'].notna()] if 'error' in df.columns else pd.DataFrame()
        if not errores.empty:
            st.warning(f"{len(errores)} keyword(s) fallaron al consultar la API.")

        df_ok = df[df['error'].isna()].copy() if 'error' in df.columns else df.copy()

        if not df_ok.empty:
            # Tendencias: una consulta por categoría única encontrada
            cat_ids_unicos = df_ok['top_category_id'].dropna().unique().tolist()
            tendencias_por_cat = {cid: obtener_tendencias(cid, headers) for cid in cat_ids_unicos}
            df_ok['en_tendencia'] = df_ok.apply(
                lambda row: row['keyword'].lower() in tendencias_por_cat.get(row['top_category_id'], []),
                axis=1
            )

            # Score de oportunidad: demanda alta + competencia baja + en tendencia
            # (evita división por cero con +1)
            df_ok['ratio_demanda_competencia'] = df_ok['demanda_vendidos_top50'] / (df_ok['competencia_publicaciones'] + 1)
            df_ok['score_oportunidad'] = (
                df_ok['ratio_demanda_competencia'] * 10
                + df_ok['en_tendencia'].astype(int) * 15
            )
            df_ok = df_ok.sort_values('score_oportunidad', ascending=False)
            st.session_state.candidatos_df = df_ok

st.divider()

if st.session_state.candidatos_df is not None:
    df_ok = st.session_state.candidatos_df
    st.subheader("📊 Resultado: Demanda vs Competencia")
    st.dataframe(
        df_ok[['keyword', 'demanda_vendidos_top50', 'competencia_publicaciones',
               'ratio_demanda_competencia', 'en_tendencia', 'precio_promedio',
               'precio_min', 'precio_max', 'score_oportunidad']],
        use_container_width=True
    )
    st.caption(
        "**ratio_demanda_competencia** alto = mucha demanda con poca oferta (buena señal). "
        "**en_tendencia** = la palabra clave aparece en el top 50 de tendencias de Mercado Libre para esa categoría."
    )

    # ==========================================================================
    # --- SECCIÓN 3 - CALCULADORA DE RENTABILIDAD (la "calculadora" oficial) ---
    # ==========================================================================
    st.header("3. Calculadora de Rentabilidad (comisión real de Mercado Libre)")
    st.caption("Usa el endpoint oficial /listing_prices para traer la comisión exacta, y le resta tu costo de producto.")

    keyword_sel = st.selectbox("Elegí una keyword analizada:", options=df_ok['keyword'].tolist())
    fila = df_ok[df_ok['keyword'] == keyword_sel].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        precio_venta = st.number_input("Precio de venta (ARS):", min_value=0.0,
                                        value=float(fila['precio_promedio'] or 0), step=100.0)
    with c2:
        costo_producto = st.number_input("Costo del producto (ARS):", min_value=0.0, value=0.0, step=100.0)
    with c3:
        peso_gramos = st.number_input("Peso facturable (gramos):", min_value=1, value=500, step=50)
    with c4:
        listing_type_id = st.selectbox("Tipo de publicación:", ["gold_special", "gold_pro", "free"])

    c5, c6 = st.columns(2)
    with c5:
        logistic_type = st.selectbox("Logística:", ["drop_off", "cross_docking", "fulfillment", "self_service"])
    with c6:
        shipping_mode = st.selectbox("Modo de envío:", ["me2", "me1", "custom"])

    if st.button("💰 Calcular Rentabilidad"):
        cat_id_calc = fila['top_category_id'] or category_id
        if not cat_id_calc:
            st.error("No hay category_id disponible para esta keyword. Elegí una categoría en la sección 2.")
        else:
            rent = calcular_rentabilidad(
                precio_venta, cat_id_calc, costo_producto, peso_gramos,
                listing_type_id, logistic_type, shipping_mode, headers
            )
            if rent.get('error'):
                st.error(f"Error consultando listing_prices: {rent['error']}")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Comisión Mercado Libre", f"${rent['comision_meli']:.2f}")
                m2.metric("Costo de envío estimado", f"${rent['costo_envio_estimado']:.2f}")
                m3.metric("Margen neto", f"${rent['margen_neto']:.2f}")
                m4.metric("Margen %", f"{rent['margen_pct']:.1f}%")
                if rent['margen_neto'] < 0:
                    st.error("⚠️ A este precio y costo, la venta da pérdida.")
                elif rent['margen_pct'] < 10:
                    st.warning("Margen bajo (<10%). Evaluá subir precio o buscar mejor costo.")
                else:
                    st.success("Margen saludable para este producto.")

                with st.expander("Ver respuesta completa de la API (listing_prices)"):
                    st.json(rent['raw'])

    st.divider()
    st.subheader("📥 Exportar")
    csv_bytes = df_ok.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar análisis completo (CSV)", data=csv_bytes,
                        file_name=f"candidatos_meli_{datetime.date.today()}.csv", mime="text/csv")

st.markdown("---")
st.caption("Buscador de Productos Candidatos v1.0 — usa /sites/{site}/search, /trends y /listing_prices de la API de Mercado Libre.")
