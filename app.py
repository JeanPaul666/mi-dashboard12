# -*- coding: utf-8 -*-
# =====================================================================================
# DASHBOARD ANALÍTICO · DATASET POS RÚSTICA (CAPA SILVER)
# Curso: Analítica Descriptiva y Diagnóstica con Visualización y Storytelling (UNMSM)
# Ejecución : streamlit run app.py
# Requisitos: pip install streamlit pandas numpy scipy plotly seaborn matplotlib pyarrow
# Archivo de datos esperado en la misma carpeta: dataset_pos_silver.parquet
#
# CAMBIOS DE ESTA VERSIÓN
#  1. La moda del ticket se reporta como CLASE MODAL (rango) + moda interpolada, no como entero.
#  2. Se elimina la variable "comensales" (datos sin calidad: hay valores > 1000).
#  3. KPI 5 y su pestaña ahora analizan Nº de TRAGOS vs Nº de PIQUEOS por precuenta:
#     correlación, asociación ("si pido trago, ¿pido piqueos?"), control de confusión,
#     secuencia temporal del pedido y efecto sobre el ticket medio.
# =====================================================================================

# ------------------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# ------------------------------------------------------------------------------------
import os                                    # Permite comprobar si el archivo parquet existe en disco
import numpy as np                           # Cálculo numérico vectorizado (arreglos, raíces, infinitos)
import pandas as pd                          # Manipulación de datos tabulares (DataFrame, groupby, pivot)
import streamlit as st                       # Framework que construye la interfaz web del dashboard
import plotly.express as px                  # Gráficos interactivos de alto nivel (barras, mapas de calor)
import plotly.graph_objects as go            # Gráficos Plotly de bajo nivel (control total de trazas)
import seaborn as sns                        # Mapa de calor de la matriz de correlación
import matplotlib.pyplot as plt              # Figura base sobre la que seaborn dibuja
from scipy import stats                      # Pruebas estadísticas (t, chi-cuadrado, correlaciones)

# ------------------------------------------------------------------------------------
# 2. CONFIGURACIÓN GENERAL DE LA PÁGINA Y ESTILO VISUAL
# ------------------------------------------------------------------------------------
st.set_page_config(                          # DEBE ser la primera instrucción de Streamlit del script
    page_title="Analítica POS Rústica",      # Título que se muestra en la pestaña del navegador
    page_icon="📊",                          # Ícono de la pestaña
    layout="wide",                           # Usa todo el ancho de la pantalla (aspecto profesional)
    initial_sidebar_state="expanded",        # La barra lateral de filtros inicia desplegada
)

RUTA_PARQUET = "dataset_pos_silver.parquet"  # Nombre del archivo local con la capa silver
COLOR_PRIMARIO = "#7A0019"                   # Color vino institucional (identidad UNMSM)
COLOR_SECUNDARIO = "#C8A24A"                 # Color dorado de acento
COLOR_NEUTRO = "#9AA3AF"                     # Gris para elementos de apoyo
DIAS_NOMBRE = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]  # Etiquetas de días (0=Lunes)
PERFILES = ["Sin tragos ni piqueos", "Solo piqueos", "Solo tragos", "Tragos + piqueos"]   # Perfiles de consumo por cuenta

st.markdown(                                 # Inyecta CSS para dar aspecto de "tarjetas" a las métricas
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}             /* Reduce márgenes de la página */
    div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E5E7EB;
        border-left: 5px solid #7A0019; padding: 12px 14px; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);}                                /* Tarjeta blanca con borde vino */
    div[data-testid="stMetricLabel"] p {font-size: 0.80rem; font-weight: 600;}  /* Etiqueta de KPI más compacta */
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}                              /* Separación entre pestañas */
    .stTabs [data-baseweb="tab"] {background: #F3F4F6; border-radius: 8px 8px 0 0; padding: 8px 14px;}
    h1, h2, h3 {color: #7A0019;}                                               /* Títulos en color vino */
    </style>
    """,
    unsafe_allow_html=True,                  # Necesario para que Streamlit acepte etiquetas HTML/CSS
)

# ------------------------------------------------------------------------------------
# 3. FUNCIONES UTILITARIAS (cálculos y formato reutilizables)
# ------------------------------------------------------------------------------------
def mostrar_tabla(d, **kw):
    """Muestra un DataFrame a todo el ancho, compatible con versiones nuevas y antiguas de Streamlit."""
    try:                                                 # Primero intenta la sintaxis moderna
        st.dataframe(d, width="stretch", **kw)           # width="stretch" = ocupar todo el ancho disponible
    except Exception:                                    # Si la versión instalada es antigua...
        st.dataframe(d, use_container_width=True, **kw)  # ...usa el parámetro clásico equivalente


def estilo(fig, alto=380):
    """Aplica un formato uniforme (plantilla limpia, márgenes, fuente) a todas las figuras Plotly."""
    fig.update_layout(template="plotly_white", height=alto,          # Fondo blanco y altura fija
                      margin=dict(l=10, r=10, t=45, b=10),           # Márgenes compactos
                      font=dict(family="Arial", size=12),            # Tipografía legible
                      legend=dict(orientation="h", y=-0.2))          # Leyenda horizontal bajo el gráfico
    return fig                                                       # Devuelve la figura ya formateada


def fmt_p(p):
    """Formatea un p-valor como se reporta en informes académicos."""
    if p is None or np.isnan(p): return "p n/d"                     # Protege contra NaN
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"             # Evita mostrar 0.0000


def fmt_num(v):
    """Número con separador de miles y sin ceros decimales sobrantes (5.0 → '5', 2.5 → '2.5')."""
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def fuerza_correlacion(r):
    """Traduce |r| a la escala estándar de interpretación vista en clase."""
    if r is None or np.isnan(r): return "No calculable"             # Protege contra valores NaN
    a = abs(r)                                                       # El signo solo indica dirección, no fuerza
    if a == 0: return "Nula"                                        # Variables totalmente independientes
    if a < 0.30: return "Débil"                                     # Rango 0.01 – 0.29
    if a < 0.50: return "Moderada"                                  # Rango 0.30 – 0.49
    if a < 0.70: return "Moderadamente fuerte"                      # Rango 0.50 – 0.69
    if a < 1.0: return "Fuerte / Alta"                              # Rango 0.70 – 0.99
    return "Perfecta"                                               # |r| = 1


def estado_semaforo(valor, meta):
    """Semáforo R/A/V: verde si cumple la meta, amarillo hasta 1.5×meta, rojo si la excede más."""
    if valor <= meta: return "🟢 Óptimo"                            # Cumple la meta
    if valor <= meta * 1.5: return "🟡 Alerta"                      # Zona de tolerancia
    return "🔴 Crítico"                                             # Fuera de tolerancia


def ic_media(x, conf):
    """Intervalo de confianza t para una media: devuelve (media, límite inferior, límite superior)."""
    x = np.asarray(x, dtype=float)                                   # Convierte a arreglo numérico
    n = len(x)                                                       # Tamaño de muestra
    m = x.mean()                                                     # Media muestral
    if n < 2: return m, np.nan, np.nan                               # Sin variabilidad no hay intervalo
    se = x.std(ddof=1) / np.sqrt(n)                                  # Error estándar = s / √n
    t_crit = stats.t.ppf(1 - (1 - conf) / 2, n - 1)                  # Valor crítico t bilateral
    return m, m - t_crit * se, m + t_crit * se                       # Media ± margen de error


def ic_diferencia_welch(a, b, conf):
    """IC de la diferencia de medias (a-b) sin asumir varianzas iguales (Welch–Satterthwaite)."""
    na, nb = len(a), len(b)                                          # Tamaños de cada grupo
    va, vb = a.var(ddof=1), b.var(ddof=1)                            # Varianzas muestrales
    se = np.sqrt(va / na + vb / nb)                                  # Error estándar de la diferencia
    gl = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))  # Grados de libertad de Welch
    t_crit = stats.t.ppf(1 - (1 - conf) / 2, gl)                     # Valor crítico t con esos grados de libertad
    dif = a.mean() - b.mean()                                        # Diferencia observada
    return dif, dif - t_crit * se, dif + t_crit * se, gl             # Diferencia, límites e información de gl


def cohen_d(a, b):
    """Tamaño del efecto d de Cohen (diferencia estandarizada con desviación combinada)."""
    na, nb = len(a), len(b)                                          # Tamaños de muestra
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))  # Desviación combinada
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan          # d = diferencia / desviación combinada


def etiqueta_efecto(d):
    """Clasifica la magnitud práctica del efecto según |d|."""
    d = abs(d)                                                       # Solo importa la magnitud
    if np.isnan(d): return "No calculable"
    if d < 0.2: return "Trivial"                                     # Prácticamente nulo
    if d < 0.5: return "Pequeño"
    if d < 0.8: return "Mediano"
    return "Grande"


def calcular_meses(pre_f):
    """Cuenta los días con datos por mes y marca como 'completo' el mes con ≥90% de sus días."""
    dias = pre_f.groupby("mes")["fecha_transaccion"].nunique()       # Días distintos con ventas por mes
    completo = pd.Series(dias.values >= 0.9 * dias.index.days_in_month, index=dias.index)  # Bandera de mes completo
    return dias, completo                                            # Devuelve ambos resultados


def tukey(s):
    """Límites de Tukey: Q1-1.5·IQR y Q3+1.5·IQR para detectar valores atípicos."""
    q1, q3 = s.quantile([0.25, 0.75]).values                         # Primer y tercer cuartil
    iqr = q3 - q1                                                    # Rango intercuartílico
    return q1, q3, iqr, q1 - 1.5 * iqr, q3 + 1.5 * iqr               # Cuartiles, IQR y límites


def corr_segura(x, y, metodo):
    """Correlación (r, p) que devuelve NaN en lugar de fallar si alguna variable es constante."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:               # Sin variabilidad no hay correlación
        return np.nan, np.nan
    f = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[metodo]
    r, p = f(x, y)
    return float(r), float(p)


# ---- Moda en RANGOS (clase modal) --------------------------------------------------
def nice_width(raw):
    """Redondea un ancho de clase calculado a un valor 'bonito' (1, 2, 5, 10, 20, 25, 50...)."""
    for c in [0.5, 1, 2, 2.5, 5, 10, 15, 20, 25, 50, 100, 200, 250, 500, 1000]:
        if c >= raw:
            return float(c)
    return float(np.ceil(raw))


def ancho_automatico(s, clases=20):
    """Ancho de clase automático: reparte hasta el percentil 99 en ~20 clases y lo redondea."""
    p99 = float(s.quantile(0.99))                                    # Se ignora la cola extrema
    return nice_width(max(p99, 1e-9) / clases)


def clase_modal(s, w):
    """Clase modal de una serie continua con clases de ancho w.
    Devuelve el rango [L, U), la moda interpolada (fórmula de Czuber) y el % de datos en la clase."""
    x = np.asarray(s, dtype=float)
    x = x[x >= 0]                                                    # El ticket válido es positivo
    if len(x) == 0:
        return None
    kmax = int(np.ceil(np.quantile(x, 0.99) / w)) + 1                # Última clase = cola (> P99), no compite por la moda
    k = np.minimum(np.floor(x / w).astype(np.int64), kmax)           # Índice de clase de cada ticket
    cnt = np.bincount(k, minlength=kmax + 1)                         # Frecuencia absoluta por clase
    j = int(cnt[:kmax].argmax())                                     # Clase con mayor frecuencia
    L = j * w                                                        # Límite inferior de la clase modal
    prev = cnt[j - 1] if j > 0 else 0                                # Frecuencia de la clase anterior
    nxt = cnt[j + 1] if j + 1 < len(cnt) else 0                      # Frecuencia de la clase siguiente
    d1, d2 = cnt[j] - prev, cnt[j] - nxt                             # Diferencias con las clases vecinas
    moda = L + w * d1 / (d1 + d2) if (d1 + d2) > 0 else L + w / 2    # Moda interpolada dentro de la clase
    return {"L": L, "U": L + w, "clase": f"[{fmt_num(L)} – {fmt_num(L + w)})",
            "moda": moda, "pct": cnt[j] / len(x) * 100}


# ------------------------------------------------------------------------------------
# 4. CARGA, LIMPIEZA Y PREPROCESAMIENTO (BACKEND, SE EJECUTA UNA SOLA VEZ Y QUEDA EN CACHÉ)
# ------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando y depurando el dataset (solo la primera vez)...")
def obtener_datos(ruta):
    """Lee el parquet, aplica limpieza idempotente y construye la tabla agregada por precuenta."""
    log = []                                                         # Bitácora de transformaciones (reproducibilidad)
    df = pd.read_parquet(ruta)                                       # Lee el archivo parquet completo en memoria
    log.append(f"Lectura inicial: {len(df):,} filas × {df.shape[1]} columnas.")  # Registra dimensiones

    df.columns = df.columns.str.strip().str.lower()                  # Estandariza nombres a minúsculas sin espacios
    requeridas = ["id_precuenta", "id_tienda", "id_cajero", "hora_precuenta", "cantidad",
                  "precio_unitario", "descuento_total", "categoria_familia"]  # Columnas imprescindibles
    faltan = [c for c in requeridas if c not in df.columns]          # Detecta columnas ausentes
    if faltan:                                                       # Si falta alguna...
        raise ValueError(f"El parquet no contiene las columnas requeridas: {faltan}")  # ...detiene con mensaje claro

    n0 = len(df)                                                     # Filas antes de deduplicar
    df = df.drop_duplicates()                                        # Elimina filas duplicadas exactas (distorsionan SUM/COUNT)
    log.append(f"Duplicados eliminados: {n0 - len(df):,}.")          # Registra cuántos se eliminaron

    for c in ["fecha_proceso", "hora_precuenta", "hora_pedido"]:     # Recorre las columnas de fecha/hora
        if c in df.columns:                                          # Solo si existen en el archivo
            df[c] = pd.to_datetime(df[c], errors="coerce")           # Convierte a datetime; lo inválido pasa a NaT
    n1 = len(df)                                                     # Filas antes de filtrar fechas inválidas
    df = df[df["hora_precuenta"].notna()]                            # Sin fecha de transacción no hay análisis temporal
    log.append(f"Filas sin fecha de precuenta descartadas: {n1 - len(df):,}.")  # Registra el descarte

    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0).astype("int32")  # Entero con signo (-1 = anulación)
    for c in ["precio_unitario", "descuento_total"]:                 # Importes decimales
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64")  # Nulos → 0.0
    log.append("Tipos numéricos normalizados (cantidad entera con signo e importes float64); nulos imputados con 0.")
    log.append("Variable 'comensales' (cant_adultos + cant_ninos) DESCARTADA del análisis: dato sin calidad "
               "(valores > 1000). Se reemplaza por Nº de tragos y Nº de piqueos por precuenta.")

    df["id_cajero"] = df["id_cajero"].astype("string").fillna("SIN_CAJERO").astype("category")  # Cajero nulo → etiqueta explícita
    df["id_tienda"] = df["id_tienda"].astype("string").astype("category")  # Tienda como categoría (ahorra memoria)
    for c in ["categoria_familia", "nombre_subcategoria"]:           # Textos con espacios laterales ('CERVEZAS ')
        if c in df.columns:                                          # Solo si la columna existe
            df[c] = df[c].astype("string").str.strip().astype("category")  # Recorta espacios y vuelve a categoría
    log.append("Textos de categoría recortados (.str.strip) y convertidos a 'category'.")

    df["fecha_transaccion"] = df["hora_precuenta"].dt.normalize()    # Fecha OFICIAL de venta (reemplaza a fecha_proceso, desfasada)
    df["total_detalle_bruto"] = df["cantidad"] * df["precio_unitario"]  # Venta teórica antes de descuentos
    df["total_detalle_neto"] = df["total_detalle_bruto"] - df["descuento_total"]  # Dinero real que cuadra con caja
    df["es_anulacion"] = df["cantidad"] < 0                          # Verdadero si la línea es una anulación posventa
    df["tiene_descuento"] = df["descuento_total"] > 0                # Verdadero si la línea tuvo descuento/promoción
    df["mes"] = df["fecha_transaccion"].dt.to_period("M").dt.to_timestamp()  # Primer día del mes (agregación temporal)
    df["dia_num"] = df["fecha_transaccion"].dt.dayofweek.astype("int8")  # 0=Lunes … 6=Domingo
    log.append("Variables derivadas creadas: fecha_transaccion, total_detalle_bruto/neto, es_anulacion, "
               "tiene_descuento, mes y dia_num.")

    g = df.groupby("id_precuenta", sort=False, observed=True)        # Agrupa las líneas por cuenta (unidad de análisis del ticket)
    pre = g.agg(                                                     # Resume cada precuenta en una sola fila
        id_tienda=("id_tienda", "first"),                            # Tienda donde ocurrió
        fecha_transaccion=("fecha_transaccion", "first"),            # Fecha de la venta
        hora_precuenta=("hora_precuenta", "first"),                  # Instante de apertura de la cuenta
        ticket_neto=("total_detalle_neto", "sum"),                   # Ticket neto = suma de líneas netas
        ticket_bruto=("total_detalle_bruto", "sum"),                 # Ticket bruto
        descuento=("descuento_total", "sum"),                        # Descuento total de la cuenta
        n_lineas=("cantidad", "size"),                               # Número de líneas de pedido
        n_anulaciones=("es_anulacion", "sum"),                       # Líneas anuladas en la cuenta
        tiene_descuento=("tiene_descuento", "max"),                  # Verdadero si alguna línea tuvo descuento
        mes=("mes", "first"),                                        # Mes contable
        dia_num=("dia_num", "first"),                                # Día 0–6
    ).reset_index()                                                  # id_precuenta vuelve a ser columna
    pre["id_tienda"] = pre["id_tienda"].astype(str)                  # Texto simple para evitar categorías vacías al agrupar
    pre["tiene_descuento"] = pre["tiene_descuento"].astype(bool)     # Garantiza tipo booleano
    pre["tipo_dia"] = np.where(pre["dia_num"] >= 5, "Sábado y Domingo", "Lunes a Viernes")  # Segmento de calendario
    pre["dia_nombre"] = pre["dia_num"].map(dict(enumerate(DIAS_NOMBRE)))  # Nombre legible del día
    log.append(f"Tabla por precuenta construida: {len(pre):,} cuentas "
               "(total_precuenta NO se suma porque se repite en cada línea).")

    calidad = pd.DataFrame({"tipo": df.dtypes.astype(str),           # Tipo de dato de cada columna
                            "nulos": df.isna().sum()})               # Nulos restantes tras la limpieza
    return df, pre, {"log": log, "calidad": calidad}                 # Entrega todo a la aplicación


@st.cache_resource(show_spinner="Clasificando tragos y piqueos por precuenta...")
def preparar_pre(tragos, piq):
    """Agrega a la tabla por precuenta: Nº de tragos, Nº de piqueos, perfil de consumo, hora y franja horaria.
    'tragos' y 'piq' son tuplas de categoria_familia elegidas por el usuario en la barra lateral."""
    df_, pre_, _ = obtener_datos(RUTA_PARQUET)                       # Dataset ya cargado
    cf = df_["categoria_familia"]                                    # Categoría de cada línea
    tmp = pd.DataFrame({                                             # Unidades netas por línea según tipo de producto
        "id_precuenta": df_["id_precuenta"].values,
        "n_tragos": np.where(cf.isin(list(tragos)), df_["cantidad"], 0),
        "n_piqueos": np.where(cf.isin(list(piq)), df_["cantidad"], 0)})
    prod = (tmp.groupby("id_precuenta", sort=False)[["n_tragos", "n_piqueos"]].sum()
               .clip(lower=0).reset_index())                         # Unidades netas por cuenta (anulaciones restan)
    out = pre_.merge(prod, on="id_precuenta", how="left")            # Une con la tabla por precuenta
    out[["n_tragos", "n_piqueos"]] = out[["n_tragos", "n_piqueos"]].fillna(0).astype("int32")
    out["tiene_trago"] = out["n_tragos"] > 0                         # ¿La cuenta incluye algún trago?
    out["tiene_piqueo"] = out["n_piqueos"] > 0                       # ¿La cuenta incluye algún piqueo?
    out["perfil"] = np.select([out["tiene_trago"] & out["tiene_piqueo"], out["tiene_trago"], out["tiene_piqueo"]],
                              ["Tragos + piqueos", "Solo tragos", "Solo piqueos"], default="Sin tragos ni piqueos")
    out["hora_dia"] = out["hora_precuenta"].dt.hour.astype("int8")   # Hora de apertura (0–23)
    out["franja"] = pd.cut(out["hora_dia"], bins=[-1, 5, 11, 17, 21, 23],
                           labels=["Madrugada (0–5h)", "Mañana (6–11h)", "Tarde (12–17h)",
                                   "Noche (18–21h)", "Noche tardía (22–23h)"]).astype(str)  # Control de confusión por horario
    return out


# ------------------------------------------------------------------------------------
# 5. FUNCIONES DE AGREGACIÓN EN CACHÉ (evitan recalcular millones de filas en cada clic)
# ------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def agg_categorias(tiendas, f1, f2):
    """Ingreso neto por tienda × categoría_familia dentro del filtro activo."""
    df_, _, _ = obtener_datos(RUTA_PARQUET)                          # Recupera el dataset ya cargado (caché de recurso)
    m = df_["id_tienda"].isin(tiendas) & df_["fecha_transaccion"].between(f1, f2)  # Máscara de filtros
    sub = df_.loc[m, ["id_tienda", "categoria_familia", "total_detalle_neto"]]     # Solo las columnas necesarias
    out = (sub.groupby(["id_tienda", "categoria_familia"], observed=True)["total_detalle_neto"]
              .sum().reset_index())                                  # GROUP BY + SUM
    return out.astype({"id_tienda": str, "categoria_familia": str})  # Texto simple para el resto del tablero


@st.cache_data(show_spinner=False)
def agg_anulaciones(tiendas, f1, f2):
    """Líneas y anulaciones por tienda × mes y por cajero."""
    df_, _, _ = obtener_datos(RUTA_PARQUET)                          # Dataset cacheado
    m = df_["id_tienda"].isin(tiendas) & df_["fecha_transaccion"].between(f1, f2)  # Máscara de filtros
    sub = df_.loc[m, ["id_tienda", "mes", "id_cajero", "es_anulacion", "total_detalle_bruto"]].copy()  # Subconjunto
    sub["monto_anulado"] = sub["total_detalle_bruto"].where(sub["es_anulacion"], 0.0)  # Importe solo de anulaciones
    por_mes = (sub.groupby(["id_tienda", "mes"], observed=True)
                  .agg(lineas=("es_anulacion", "size"), anulaciones=("es_anulacion", "sum"),
                       monto_anulado=("monto_anulado", "sum")).reset_index())  # COUNT y SUM por tienda-mes
    por_cajero = (sub.groupby("id_cajero", observed=True)
                     .agg(lineas=("es_anulacion", "size"), anulaciones=("es_anulacion", "sum"))
                     .reset_index())                                 # COUNT y SUM por cajero
    por_mes["id_tienda"] = por_mes["id_tienda"].astype(str)          # Texto simple
    por_cajero["id_cajero"] = por_cajero["id_cajero"].astype(str)    # Texto simple
    return por_mes, por_cajero                                       # Devuelve ambas tablas


@st.cache_data(show_spinner="Calculando correlaciones...")
def corr_resultados(tiendas, f1, f2, tragos, piq, solo_consumo):
    """Pearson, Spearman y Kendall entre Nº de tragos y Nº de piqueos, más matrices y controles."""
    pre_ = preparar_pre(tragos, piq)                                 # Tabla por precuenta con tragos/piqueos
    m = (pre_["id_tienda"].isin(tiendas) & pre_["fecha_transaccion"].between(f1, f2)
         & (pre_["ticket_neto"] > 0))                                # Cuentas válidas: ticket positivo
    if solo_consumo:                                                 # Opción: solo cuentas con ≥1 trago o piqueo
        m &= (pre_["n_tragos"] + pre_["n_piqueos"]) > 0
    b = pre_.loc[m, ["id_tienda", "n_tragos", "n_piqueos", "n_lineas", "descuento", "hora_dia", "ticket_neto"]].copy()
    out = {"n": len(b)}                                              # Tamaño de la base de correlación
    if len(b) < 30:                                                  # Con menos de 30 cuentas no es confiable
        return out
    if b["n_tragos"].nunique() < 2 or b["n_piqueos"].nunique() < 2:  # Una variable constante no correlaciona
        out["sin_varianza"] = True
        return out
    vars_ = ["n_tragos", "n_piqueos", "n_lineas", "descuento", "hora_dia", "ticket_neto"]  # Variables de la matriz
    muestra = b.sample(min(20000, len(b)), random_state=42)          # Muestra reproducible para Kendall (costoso)
    out["pearson_m"] = b[vars_].corr(method="pearson")               # Matriz de Pearson (relación lineal)
    out["spearman_m"] = b[vars_].corr(method="spearman")             # Matriz de Spearman (relación monótona)
    out["kendall_m"] = muestra[vars_].corr(method="kendall")         # Matriz de Kendall (sobre la muestra)
    xt, xp, yt = (b["n_tragos"].to_numpy(float), b["n_piqueos"].to_numpy(float), b["ticket_neto"].to_numpy(float))
    out["pearson_r"], out["pearson_p"] = corr_segura(xt, xp, "pearson")      # Tragos vs piqueos
    out["spearman_rho"], out["spearman_p"] = corr_segura(xt, xp, "spearman")
    out["kendall_tau"], out["kendall_p"] = corr_segura(muestra["n_tragos"], muestra["n_piqueos"], "kendall")
    out["rho_tragos_ticket"] = corr_segura(xt, yt, "spearman")[0]    # Cada producto vs ticket
    out["rho_piqueos_ticket"] = corr_segura(xp, yt, "spearman")[0]
    q_t, q_p = b["n_tragos"].quantile(0.99), b["n_piqueos"].quantile(0.99)   # Cortes de atípicos (P99)
    sin = b[(b["n_tragos"] <= q_t) & (b["n_piqueos"] <= q_p)]        # Base sin valores extremos
    out["pearson_sin_outliers"] = corr_segura(sin["n_tragos"], sin["n_piqueos"], "pearson")[0]
    out["por_tienda"] = pd.Series({t: corr_segura(g["n_tragos"], g["n_piqueos"], "spearman")[0]
                                   for t, g in b.groupby("id_tienda") if len(g) >= 30})  # Control por tienda
    out["cruce"] = pd.crosstab(b["n_tragos"].clip(upper=10), b["n_piqueos"].clip(upper=10))  # Tabla tragos × piqueos
    return out


@st.cache_data(show_spinner="Ajustando modelo de ticket...")
def modelo_ticket(tiendas, f1, f2, tragos, piq):
    """Regresión OLS: ticket ~ tragos + piqueos + tragos×piqueos (ticket recortado en P99)."""
    pre_ = preparar_pre(tragos, piq)
    m = (pre_["id_tienda"].isin(tiendas) & pre_["fecha_transaccion"].between(f1, f2)
         & (pre_["ticket_neto"] > 0))
    b = pre_.loc[m, ["n_tragos", "n_piqueos", "ticket_neto"]]
    if len(b) < 100:
        return None
    if len(b) > 300000:                                              # Muestra reproducible para velocidad
        b = b.sample(300000, random_state=7)
    y = b["ticket_neto"].clip(upper=b["ticket_neto"].quantile(0.99)).to_numpy(float)
    t, p = b["n_tragos"].to_numpy(float), b["n_piqueos"].to_numpy(float)
    if t.std() == 0 or p.std() == 0:
        return None
    X = np.column_stack([np.ones(len(b)), t, p, t * p])              # Intercepto, tragos, piqueos, interacción
    try:
        XtX = X.T @ X
        beta = np.linalg.solve(XtX, X.T @ y)                         # Mínimos cuadrados ordinarios
        res = y - X @ beta
        n, k = X.shape
        s2 = res @ res / (n - k)                                     # Varianza residual
        se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))               # Errores estándar
    except np.linalg.LinAlgError:
        return None
    tv = beta / se
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    tabla = pd.DataFrame({"Término": ["Intercepto", "Por cada trago", "Por cada piqueo", "Interacción trago × piqueo"],
                          "Coeficiente": beta, "Error est.": se, "t": tv,
                          "p": 2 * stats.norm.sf(np.abs(tv))})
    return {"tabla": tabla, "r2": r2, "n": n}


@st.cache_data(show_spinner=False)
def secuencia_pedido(tiendas, f1, f2, tragos, piq):
    """En cuentas con ambos tipos de producto: ¿qué se pidió primero, trago o piqueo? (usa hora_pedido)."""
    df_, _, _ = obtener_datos(RUTA_PARQUET)
    if "hora_pedido" not in df_.columns:
        return None
    m = (df_["id_tienda"].isin(tiendas) & df_["fecha_transaccion"].between(f1, f2)
         & (df_["cantidad"] > 0) & df_["hora_pedido"].notna())
    sub = df_.loc[m, ["id_precuenta", "categoria_familia", "hora_pedido"]]
    es_t = sub["categoria_familia"].isin(list(tragos)).to_numpy()
    es_p = sub["categoria_familia"].isin(list(piq)).to_numpy()
    sub = sub.assign(tipo=np.where(es_t, "t", np.where(es_p, "p", "")))
    sub = sub[sub["tipo"] != ""]
    g = sub.groupby(["id_precuenta", "tipo"])["hora_pedido"].min().unstack()  # Primer pedido de cada tipo por cuenta
    if "t" not in g.columns or "p" not in g.columns:
        return None
    g = g.dropna()                                                   # Solo cuentas con ambos tipos
    if len(g) == 0:
        return None
    return {"n": len(g), "trago_primero": int((g["t"] < g["p"]).sum()),
            "piqueo_primero": int((g["p"] < g["t"]).sum()), "simultaneo": int((g["t"] == g["p"]).sum())}


def test_brecha(pre_f, conf):
    """Compara el ticket neto de Sábado y Domingo vs Lunes a Viernes con t de Welch y Mann-Whitney."""
    pv = pre_f[pre_f["ticket_neto"] > 0]                             # Solo cuentas con ticket positivo
    a = pv.loc[pv["tipo_dia"] == "Sábado y Domingo", "ticket_neto"].values  # Grupo A: fin de jornada de descanso
    b = pv.loc[pv["tipo_dia"] == "Lunes a Viernes", "ticket_neto"].values   # Grupo B: días hábiles
    if len(a) < 2 or len(b) < 2:                                     # Se necesitan ambos grupos con datos
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)                    # t de Welch (no asume varianzas iguales)
    dif, lo, hi, gl = ic_diferencia_welch(a, b, conf)                # Diferencia con su intervalo de confianza
    rng = np.random.default_rng(42)                                  # Generador reproducible
    sa = rng.choice(a, size=min(len(a), 100000), replace=False)      # Submuestra A para la prueba no paramétrica
    sb = rng.choice(b, size=min(len(b), 100000), replace=False)      # Submuestra B
    _, p_u = stats.mannwhitneyu(sa, sb, alternative="two-sided")     # U de Mann-Whitney (alternativa robusta a outliers)
    return {"media_a": a.mean(), "media_b": b.mean(), "n_a": len(a), "n_b": len(b),
            "brecha_pct": (a.mean() / b.mean() - 1) * 100, "t": t, "p": p, "gl": gl,
            "dif": dif, "lo": lo, "hi": hi, "d": cohen_d(a, b), "p_u": p_u,
            "ic_a": ic_media(a, conf), "ic_b": ic_media(b, conf)}    # Todo el resultado en un diccionario


# ------------------------------------------------------------------------------------
# 6. CARGA DE DATOS Y VALIDACIÓN DE EXISTENCIA DEL ARCHIVO
# ------------------------------------------------------------------------------------
if not os.path.exists(RUTA_PARQUET):                                 # Verifica que el parquet esté junto a app.py
    st.error(f"No se encontró '{RUTA_PARQUET}'. Colócalo en la misma carpeta que app.py y recarga.")
    st.stop()                                                        # Detiene la ejecución del script

try:                                                                 # Intenta cargar y limpiar los datos
    df, pre_base, info = obtener_datos(RUTA_PARQUET)                 # Ejecuta (o recupera de caché) el preprocesamiento
except Exception as e:                                               # Si ocurre cualquier problema...
    st.error(f"Error al preparar los datos: {e}")                    # ...lo muestra de forma legible
    st.stop()                                                        # ...y detiene la aplicación

# ------------------------------------------------------------------------------------
# 7. BARRA LATERAL: FILTROS Y PARÁMETROS ESTADÍSTICOS
# ------------------------------------------------------------------------------------
with st.sidebar:                                                     # Todo lo que sigue se dibuja en la barra lateral
    st.markdown("## 🎛️ Panel de control")                            # Título del panel
    tiendas_todas = sorted(map(str, df["id_tienda"].unique()))       # Lista ordenada de tiendas disponibles
    tiendas_sel = st.multiselect("Tiendas", tiendas_todas, default=tiendas_todas)  # Filtro multiselección
    fmin = pre_base["fecha_transaccion"].min().date()                # Primera fecha del dataset
    fmax = pre_base["fecha_transaccion"].max().date()                # Última fecha del dataset
    rango = st.date_input("Rango de fechas de venta", value=(fmin, fmax), min_value=fmin, max_value=fmax)  # Calendario
    st.markdown("---")                                               # Separador visual
    st.markdown("### Variables de consumo (tragos / piqueos)")       # Define qué categorías cuentan como cada tipo
    cats_all = sorted(df["categoria_familia"].dropna().astype(str).unique())  # Categorías existentes en la data
    def_t = [c for c in cats_all if "TRAGO" in c.upper()]            # Detección automática: contiene "TRAGO"
    def_p = [c for c in cats_all if "PIQUEO" in c.upper()]           # Detección automática: contiene "PIQUEO"
    cats_trago = st.multiselect("Categorías que cuentan como TRAGOS", cats_all, default=def_t,
                                help="Se detectan automáticamente las que contienen 'TRAGO'. Agrega o quita las que correspondan.")
    cats_piq = st.multiselect("Categorías que cuentan como PIQUEOS", cats_all, default=def_p,
                              help="Se detectan automáticamente las que contienen 'PIQUEO'. Agrega o quita las que correspondan.")
    solapadas = set(cats_trago) & set(cats_piq)                      # Una categoría no puede ser trago y piqueo a la vez
    if solapadas:
        st.warning(f"Categorías en ambas listas (se dejan como tragos): {', '.join(sorted(solapadas))}")
        cats_piq = [c for c in cats_piq if c not in solapadas]
    base_corr = st.radio("Cuentas incluidas en la correlación",
                         ["Todas las cuentas válidas", "Solo cuentas con ≥1 trago o piqueo"],
                         help="La segunda opción evita que las cuentas sin tragos ni piqueos (0,0) inflen la relación.")
    st.markdown("---")
    st.markdown("### Parámetros de inferencia")                      # Subtítulo
    alpha = st.selectbox("Nivel de significancia (α)", [0.05, 0.01, 0.001], index=2,
                         help="Con millones de registros se recomienda α = 0.001 y reportar el tamaño del efecto.")
    conf = st.selectbox("Nivel de confianza del IC", [0.90, 0.95, 0.99], index=1)  # Confianza de los intervalos
    st.markdown("### Parámetros de KPI")                             # Subtítulo
    meta_anul = st.number_input("Meta de anulación (% de líneas)", 0.1, 20.0, 2.0, 0.1)  # Meta editable del KPI 3
    umbral_caida = st.slider("Umbral de caída mensual (%)", 1, 40, 10)  # Umbral de alerta del KPI 4
    st.caption("Los valores por defecto son supuestos de trabajo: ajústalos a la meta real del negocio.")

if len(tiendas_sel) == 0:                                            # Si el usuario deseleccionó todas las tiendas
    st.warning("Selecciona al menos una tienda.")
    st.stop()
if not isinstance(rango, (tuple, list)) or len(rango) != 2:          # El calendario requiere fecha inicial y final
    st.info("Selecciona una fecha inicial y una final.")
    st.stop()

tiendas_sel = tuple(tiendas_sel)                                     # Tupla (hashable) para las funciones en caché
cats_trago, cats_piq = tuple(sorted(cats_trago)), tuple(sorted(cats_piq))  # Tuplas hashables para la caché
solo_consumo = base_corr.startswith("Solo")                          # Bandera de la base de correlación
f1, f2 = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])              # Fechas del filtro como Timestamp
pre = preparar_pre(cats_trago, cats_piq)                             # Tabla por precuenta + tragos/piqueos
pre_f = pre[pre["id_tienda"].isin(tiendas_sel) & pre["fecha_transaccion"].between(f1, f2)]  # Precuentas filtradas
if pre_f.empty:                                                      # Sin datos no hay nada que graficar
    st.warning("No hay ventas con los filtros seleccionados.")
    st.stop()

# ------------------------------------------------------------------------------------
# 8. ENCABEZADO Y CÁLCULO DE LOS 6 KPIs PRINCIPALES
# ------------------------------------------------------------------------------------
st.title("📊 Analítica Descriptiva y Diagnóstica · POS Rústica")     # Título principal
st.caption(f"{len(pre_f):,} precuentas · {len(tiendas_sel)} tiendas · "
           f"{f1:%Y-%m-%d} a {f2:%Y-%m-%d}")                         # Subtítulo con el alcance del filtro

pv = pre_f[pre_f["ticket_neto"] > 0]                                 # Precuentas válidas (ticket positivo)
mediana_g = pv["ticket_neto"].median()                               # KPI 1: ticket neto mediano
cv_g = pv["ticket_neto"].std() / pv["ticket_neto"].mean() * 100      # Coeficiente de variación global

cat_f = agg_categorias(tiendas_sel, f1, f2)                          # Ingreso por tienda × categoría
tot_cat = cat_f.groupby("categoria_familia")["total_detalle_neto"].sum().sort_values(ascending=False)  # Ingreso por categoría
top3_g = tot_cat.head(3).sum() / tot_cat.sum() * 100                 # KPI 2: % del ingreso en las 3 categorías líderes

por_mes_an, por_cajero_an = agg_anulaciones(tiendas_sel, f1, f2)     # Datos de anulación
tasa_g = por_mes_an["anulaciones"].sum() / por_mes_an["lineas"].sum() * 100  # KPI 3: tasa global de anulación

dias_datos, completo = calcular_meses(pre_f)                         # Meses completos vs parciales
tot_mes = pre_f.groupby("mes")["ticket_neto"].sum()                  # Ingreso neto mensual total
tot_ok = tot_mes.where(completo.reindex(tot_mes.index))              # Anula meses incompletos (evita falsas caídas)
mom_tot = ((tot_ok / tot_ok.shift(1) - 1) * 100).replace([np.inf, -np.inf], np.nan).dropna()  # Variación mensual %
mom_ult = mom_tot.iloc[-1] if len(mom_tot) else np.nan               # KPI 4: último crecimiento mensual válido

cr = corr_resultados(tiendas_sel, f1, f2, cats_trago, cats_piq, solo_consumo)  # KPI 5: resultados de correlación
rho = cr.get("spearman_rho", np.nan)                                 # ρ de Spearman (NaN si no hay datos suficientes)

res_b = test_brecha(pre_f, conf)                                     # KPI 6: prueba de brecha de ticket

k1, k2, k3, k4, k5, k6 = st.columns(6)                               # Seis columnas para seis KPIs
k1.metric("Ticket neto mediano", f"{mediana_g:,.2f}", f"CV {cv_g:.0f}%", delta_color="off")
k2.metric("Concentración Top-3 categorías", f"{top3_g:.1f}%", "del ingreso neto", delta_color="off")
k3.metric("Tasa de anulación", f"{tasa_g:.2f}%", f"meta ≤ {meta_anul:.1f}%",
          delta_color="inverse" if tasa_g > meta_anul else "normal")
k4.metric("Crecimiento mensual del ingreso",
          "n/d" if np.isnan(mom_ult) else f"{mom_ult:+.1f}%",
          "" if np.isnan(mom_ult) else f"mes {mom_tot.index[-1]:%Y-%m}", delta_color="off")
k5.metric("Correlación ρ tragos–piqueos", "n/d" if np.isnan(rho) else f"{rho:.3f}",
          "" if np.isnan(rho) else fuerza_correlacion(rho), delta_color="off")
if res_b:                                                            # Solo si la prueba pudo calcularse
    k6.metric("Brecha ticket Sáb-Dom vs Lun-Vie", f"{res_b['brecha_pct']:+.1f}%",
              "significativa" if res_b["p"] <= alpha else "no significativa", delta_color="off")
else:
    k6.metric("Brecha ticket Sáb-Dom vs Lun-Vie", "n/d", "sin datos en ambos grupos", delta_color="off")

with st.expander("🧹 Bitácora de preprocesamiento (backend) y calidad de datos"):  # Panel plegable de auditoría
    for linea in info["log"]:                                        # Muestra cada paso de limpieza registrado
        st.write("•", linea)
    mostrar_tabla(info["calidad"])                                   # Tipos de dato y nulos finales por columna

# ------------------------------------------------------------------------------------
# 9. PESTAÑAS TEMÁTICAS (una por KPI, nombradas con el tema técnico)
# ------------------------------------------------------------------------------------
tabs = st.tabs([
    "📈 Ticket Neto · Tendencia Central y Dispersión",
    "🧩 Mix de Carta · Segmentación y Cruces",
    "🚦 Tasa de Anulación · Semáforo de KPI",
    "💰 Ingreso Neto · Agregación y Crecimiento",
    "🔗 Tragos vs Piqueos · Correlación y Asociación",
    "🧪 Brecha de Ticket · Pruebas de Hipótesis",
])

# ===================== PESTAÑA 1 · ESTADÍSTICA DESCRIPTIVA =====================
with tabs[0]:
    st.subheader("KPI 1 · Ticket Neto Mediano por Precuenta")
    st.info("**Pregunta SMART:** ¿Cuál es el ticket neto mediano por precuenta en cada tienda durante el periodo "
            "filtrado y cuántas tiendas superan un CV del 35%, para priorizar en el próximo trimestre la "
            "estandarización de la carta y los precios en las tiendas más heterogéneas?")
    st.caption(f"Se excluyen {len(pre_f) - len(pv):,} precuentas con ticket ≤ 0 (anulaciones totales). "
               "Se usa la mediana como KPI porque la distribución del ticket es asimétrica.")

    auto_w = ancho_automatico(pv["ticket_neto"])                     # Ancho de clase sugerido
    ancho_user = st.number_input("Ancho de clase para la moda (0 = automático)", min_value=0.0, max_value=1000.0,
                                 value=0.0, step=0.5,
                                 help=f"Con 0 se usa un ancho automático de {fmt_num(auto_w)} (≈20 clases hasta el P99).")
    w_clase = ancho_user if ancho_user > 0 else auto_w               # Ancho de clase efectivo
    mod_g = clase_modal(pv["ticket_neto"], w_clase)                  # Clase modal global

    def resumen_serie(s):                                            # Calcula todas las medidas descriptivas de una serie
        q1, q3, iqr, li, ls = tukey(s)                               # Cuartiles y límites de Tukey
        media, desv = s.mean(), s.std()                              # Media y desviación estándar muestral
        return pd.Series({
            "n": len(s), "media": media, "mediana": s.median(),      # Tendencia central
            "desv_est": desv, "cv_pct": desv / media * 100,          # Dispersión absoluta y relativa
            "q1": q1, "q3": q3, "iqr": iqr, "p90": s.quantile(0.9),  # Posición
            "min": s.min(), "max": s.max(),                          # Extremos
            "asimetria": s.skew(), "curtosis": s.kurt(),             # Forma de la distribución
            "pct_outliers": ((s < li) | (s > ls)).mean() * 100,      # % de valores atípicos (Tukey)
            "bigote_inf": s[s >= li].min(), "bigote_sup": s[s <= ls].max()})  # Extremos del boxplot

    tabla = pv.groupby("id_tienda")["ticket_neto"].apply(resumen_serie).unstack()  # Resumen numérico por tienda
    tabla.loc["TOTAL (selección)"] = resumen_serie(pv["ticket_neto"])  # Fila global
    modales = {t: clase_modal(g["ticket_neto"], w_clase) for t, g in pv.groupby("id_tienda")}  # Clase modal por tienda
    modales["TOTAL (selección)"] = mod_g
    tabla["clase_modal"] = pd.Series({t: (r["clase"] if r else "n/d") for t, r in modales.items()})  # Rango modal
    tabla["moda_interp"] = pd.Series({t: (r["moda"] if r else np.nan) for t, r in modales.items()})  # Moda interpolada
    tabla["pct_clase_modal"] = pd.Series({t: (r["pct"] if r else np.nan) for t, r in modales.items()})  # % de cuentas en la clase
    cols_vista = ["n", "media", "mediana", "clase_modal", "moda_interp", "pct_clase_modal", "desv_est", "cv_pct",
                  "q1", "q3", "iqr", "p90", "min", "max", "asimetria", "curtosis", "pct_outliers"]  # Columnas visibles
    mostrar_tabla(tabla[cols_vista].round(2))                        # Tabla de estadísticos descriptivos
    st.caption(f"**Moda en rangos:** 'clase_modal' es el intervalo [inferior – superior) de ancho {fmt_num(w_clase)} "
               "con más precuentas; 'moda_interp' es la moda estimada dentro de ese intervalo (fórmula de Czuber) y "
               "'pct_clase_modal' el % de cuentas que caen en él. Todas las tiendas usan el mismo ancho para ser comparables.")

    sesgo = "positivo (cola larga a la derecha: la media supera a la mediana)" \
        if pv["ticket_neto"].mean() > pv["ticket_neto"].median() else "nulo o negativo"
    n_cv = int((tabla.drop(index="TOTAL (selección)")["cv_pct"].astype(float) > 35).sum())  # Tiendas con CV > 35%
    st.markdown(f"**Lectura:** sesgo {sesgo}. **{n_cv}** de {len(tabla) - 1} tiendas presentan CV > 35% "
                "(alta heterogeneidad del ticket).")

    c1, c2 = st.columns(2)                                           # Dos columnas de gráficos
    with c1:
        tope = pv["ticket_neto"].quantile(0.99)                      # Se recorta en P99 para ver bien la forma
        conteo, bordes = np.histogram(pv["ticket_neto"].clip(upper=tope), bins=60)  # Histograma numérico
        centros = (bordes[:-1] + bordes[1:]) / 2                     # Punto medio de cada intervalo
        fig = go.Figure(go.Bar(x=centros, y=conteo, marker_color=COLOR_PRIMARIO, name="Histograma"))  # Barras
        fig.add_trace(go.Scatter(x=centros, y=conteo, mode="lines", line=dict(color=COLOR_SECUNDARIO, width=3),
                                 name="Polígono de frecuencias"))    # Línea que une los puntos medios
        if mod_g:                                                    # Sombrea la clase modal global
            fig.add_vrect(x0=mod_g["L"], x1=min(mod_g["U"], tope), fillcolor=COLOR_SECUNDARIO, opacity=0.25,
                          line_width=0, annotation_text=f"Clase modal {mod_g['clase']}", annotation_position="top left")
        fig.add_vline(x=pv["ticket_neto"].mean(), line_dash="dash", line_color="black", annotation_text="Media")
        fig.add_vline(x=pv["ticket_neto"].median(), line_dash="dot", line_color="green", annotation_text="Mediana")
        fig.update_layout(title="Distribución del ticket neto (recortada en P99)", xaxis_title="Ticket neto",
                          yaxis_title="Precuentas")
        st.plotly_chart(estilo(fig))                                 # Dibuja el gráfico en Streamlit
    with c2:
        fig = go.Figure()                                            # Boxplot construido con cuartiles precalculados
        for t, fila in tabla.drop(index="TOTAL (selección)").iterrows():  # Una caja por tienda
            fig.add_trace(go.Box(x=[t], q1=[float(fila["q1"])], median=[float(fila["mediana"])], q3=[float(fila["q3"])],
                                 lowerfence=[float(fila["bigote_inf"])], upperfence=[float(fila["bigote_sup"])],
                                 mean=[float(fila["media"])], marker_color=COLOR_PRIMARIO, showlegend=False))
        fig.update_layout(title="Diagrama de caja por tienda (sin atípicos)", yaxis_title="Ticket neto")
        st.plotly_chart(estilo(fig))

    c3, c4 = st.columns(2)
    with c3:
        cv_df = tabla.drop(index="TOTAL (selección)")["cv_pct"].astype(float).reset_index()  # CV por tienda
        cv_df.columns = ["tienda", "cv"]                             # Nombres claros
        cv_df["nivel"] = pd.cut(cv_df["cv"], [-np.inf, 15, 35, np.inf],
                                labels=["Baja (<15%)", "Moderada (15–35%)", "Alta (>35%)"])  # Regla práctica del CV
        fig = px.bar(cv_df, x="tienda", y="cv", color="nivel",
                     color_discrete_map={"Baja (<15%)": "#2E8B57", "Moderada (15–35%)": COLOR_SECUNDARIO,
                                         "Alta (>35%)": COLOR_PRIMARIO},
                     title="Coeficiente de variación del ticket por tienda")
        fig.add_hline(y=35, line_dash="dash", annotation_text="Umbral 35%")
        st.plotly_chart(estilo(fig))
    with c4:
        bordes_f = np.unique(np.round(pv["ticket_neto"].quantile([0, .2, .4, .6, .8, .95, 1]).values, 2))  # Cortes por cuantiles
        fr = pd.cut(pv["ticket_neto"], bins=bordes_f, include_lowest=True).value_counts().sort_index()  # Frecuencia absoluta
        tf = pd.DataFrame({"Intervalo de ticket": fr.index.astype(str), "Frec. absoluta": fr.values})  # Tabla base
        tf["Frec. relativa (%)"] = (tf["Frec. absoluta"] / tf["Frec. absoluta"].sum() * 100).round(2)  # Relativa
        tf["Frec. acumulada (%)"] = tf["Frec. relativa (%)"].cumsum().round(2)  # Acumulada
        st.markdown("**Distribución de frecuencias del ticket**")
        mostrar_tabla(tf)
        _, _, iqr_g, li_g, ls_g = tukey(pv["ticket_neto"])           # Límites globales de Tukey
        st.caption(f"Método IQR (Tukey): IQR = {iqr_g:,.2f}; atípico si ticket < {li_g:,.2f} o > {ls_g:,.2f}. "
                   "No se eliminan: se investigan y se reportan análisis con y sin ellos.")

# ===================== PESTAÑA 2 · SEGMENTACIÓN, AGRUPACIÓN Y CRUCES =====================
with tabs[1]:
    st.subheader("KPI 2 · Concentración del Ingreso en las 3 Categorías Líderes")
    st.info("**Pregunta SMART:** ¿Qué porcentaje del ingreso neto de cada tienda proviene de sus 3 categorías "
            "principales y en qué cuartil de ingreso se ubica cada tienda, para focalizar el mix de carta y las "
            "promociones del próximo trimestre?")
    ing_t = cat_f.groupby("id_tienda")["total_detalle_neto"].sum()   # Ingreso neto por tienda
    top3_pct = cat_f.groupby("id_tienda")["total_detalle_neto"].apply(
        lambda s: s.nlargest(3).sum() / s.sum() * 100)               # % del ingreso de las 3 primeras categorías
    top_filas = cat_f.sort_values("total_detalle_neto", ascending=False).groupby("id_tienda").head(3)  # Top-3 por tienda
    nombres_top = top_filas.groupby("id_tienda")["categoria_familia"].agg(lambda s: ", ".join(s))  # Nombres de las 3
    seg = pd.DataFrame({"Ingreso neto": ing_t, "Participación (%)": ing_t / ing_t.sum() * 100,
                        "Top-3 categorías (%)": top3_pct, "Categorías líderes": nombres_top})  # Tabla integrada
    if len(ing_t) >= 4:                                              # Los cuartiles requieren al menos 4 tiendas
        seg["Cuartil de ingreso"] = pd.qcut(ing_t.rank(method="first"), 4,
                                            labels=["Q1 · Bajo", "Q2 · Medio-bajo", "Q3 · Medio-alto", "Q4 · Alto"])
    else:
        st.caption("Se necesitan ≥ 4 tiendas seleccionadas para segmentar por cuartiles.")
    mostrar_tabla(seg.sort_values("Ingreso neto", ascending=False).round(2))  # Tabla ordenada por ingreso

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(top3_pct.reset_index(), x="id_tienda", y="total_detalle_neto",
                     labels={"id_tienda": "Tienda", "total_detalle_neto": "% del ingreso en Top-3"},
                     title="Concentración Top-3 de categorías por tienda", color_discrete_sequence=[COLOR_PRIMARIO])
        st.plotly_chart(estilo(fig))
    with c2:
        par = tot_cat.reset_index()                                  # Diagrama de Pareto de categorías
        par.columns = ["categoria", "ingreso"]
        par = par.head(15)                                           # 15 categorías principales
        par["acum"] = par["ingreso"].cumsum() / tot_cat.sum() * 100  # % acumulado sobre el total
        fig = go.Figure(go.Bar(x=par["categoria"], y=par["ingreso"], marker_color=COLOR_PRIMARIO, name="Ingreso"))
        fig.add_trace(go.Scatter(x=par["categoria"], y=par["acum"], yaxis="y2", mode="lines+markers",
                                 line=dict(color=COLOR_SECUNDARIO), name="% acumulado"))
        fig.update_layout(title="Pareto de categorías (Top 15)",
                          yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="% acumulado"))
        st.plotly_chart(estilo(fig))

    piv = cat_f.pivot_table(index="id_tienda", columns="categoria_familia", values="total_detalle_neto",
                            aggfunc="sum", fill_value=0)             # Tabla dinámica tienda × categoría
    piv_pct = piv.div(piv.sum(axis=1), axis=0) * 100                 # % por fila (composición del ingreso por tienda)
    fig = px.imshow(piv_pct, aspect="auto", color_continuous_scale="Reds",
                    labels=dict(color="% del ingreso"), title="Tabla dinámica: composición del ingreso por tienda × categoría")
    st.plotly_chart(estilo(fig, 430))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Segmentación por desviación estándar (μ ± σ) del ticket**")
        mu, sd = pv["ticket_neto"].mean(), pv["ticket_neto"].std()   # Media y desviación del ticket
        cortes = [-np.inf, mu - sd, mu, mu + sd, mu + 2 * sd, np.inf]  # Fronteras de los segmentos
        etiquetas = ["Crítico (< μ−1σ)", "Riesgo [μ−1σ, μ)", "Esperado [μ, μ+1σ)",
                     "Alta demanda [μ+1σ, μ+2σ)", "Saturado (≥ μ+2σ)"]  # Nombres diagnósticos
        sg = pd.cut(pv["ticket_neto"], bins=cortes, labels=etiquetas, right=False).value_counts().reindex(etiquetas)
        sg = pd.DataFrame({"Segmento": sg.index, "Precuentas": sg.values,
                           "%": (sg.values / sg.sum() * 100).round(2)})  # Tabla de resultados
        mostrar_tabla(sg)
        st.caption("Como el ticket no puede ser negativo, el segmento 'Crítico' puede quedar vacío (límite natural 0).")
    with c4:
        st.markdown("**Rangos personalizados de ticket**")
        txt = st.text_input("Límites separados por coma", "0,5,10,20,40,80")  # El usuario define sus propios cortes
        try:
            lims = sorted(set(float(x) for x in txt.split(",") if x.strip()))  # Convierte el texto a números únicos ordenados
            rp = pd.cut(pv["ticket_neto"], bins=lims + [np.inf], right=False).value_counts().sort_index()
            rp = pd.DataFrame({"Rango": rp.index.astype(str), "Precuentas": rp.values,
                               "%": (rp.values / rp.sum() * 100).round(2)})
            mostrar_tabla(rp)
        except ValueError:
            st.warning("Escribe números separados por coma, por ejemplo: 0,5,10,20")

    st.markdown("**Tabla cruzada (cross-tabulation): tienda × precuentas con descuento**")
    ct = pd.crosstab(pre_f["id_tienda"], pre_f["tiene_descuento"].map({True: "Con descuento", False: "Sin descuento"}),
                     margins=True, margins_name="Total")             # Frecuencias absolutas con totales
    ct_pct = (pd.crosstab(pre_f["id_tienda"], pre_f["tiene_descuento"].map({True: "Con descuento", False: "Sin descuento"}),
                          normalize="index") * 100).round(2)         # % por fila
    cc1, cc2 = st.columns(2)
    with cc1:
        mostrar_tabla(ct)
    with cc2:
        mostrar_tabla(ct_pct.add_suffix(" (%)"))
    ing_td = pre_f.pivot_table(index="id_tienda", columns="tipo_dia", values="ticket_neto", aggfunc="sum")  # Agrupación múltiple
    fig = px.bar(ing_td.reset_index().melt(id_vars="id_tienda", var_name="Tipo de día", value_name="Ingreso neto"),
                 x="id_tienda", y="Ingreso neto", color="Tipo de día", barmode="group",
                 color_discrete_sequence=[COLOR_PRIMARIO, COLOR_SECUNDARIO],
                 title="Agrupación múltiple: ingreso neto por tienda y tipo de día")
    st.plotly_chart(estilo(fig))

# ===================== PESTAÑA 3 · KPI CON FICHA TÉCNICA Y SEMÁFORO =====================
with tabs[2]:
    st.subheader("KPI 3 · Tasa de Anulación Posventa (%)")
    pregunta3 = (f"¿Qué porcentaje de las líneas de pedido de cada tienda corresponde a anulaciones (cantidad < 0) "
                 f"cada mes y qué tiendas superan la meta de {meta_anul:.1f}%, para activar una auditoría de "
                 "cajeros y procedimientos de anulación durante el mes siguiente?")
    st.info("**Pregunta SMART:** " + pregunta3)
    with st.expander("📋 Ficha técnica estándar del KPI (The KPI Institute)", expanded=False):
        ficha = pd.DataFrame([
            ("1. Identificación", "ID del KPI", "KPI-OPE-003"),
            ("1. Identificación", "Nombre del KPI", "% Tasa de Anulación Posventa"),
            ("2. Definición y Uso", "Definición conceptual", "Porcentaje de líneas de pedido con cantidad negativa sobre el total de líneas."),
            ("2. Definición y Uso", "Propósito estratégico", "Detectar fuga de ingresos y riesgo operativo en caja."),
            ("2. Definición y Uso", "Pregunta clave (KPQ) – SMART", pregunta3),
            ("3. Métodos de cálculo", "Fórmula matemática", "(Líneas con cantidad < 0 / Total de líneas) × 100"),
            ("3. Métodos de cálculo", "Unidad de medida", "Porcentaje (%)"),
            ("4. Enfoque e impacto", "Tipo de indicador", "Lagging (histórico) / Outcome"),
            ("4. Enfoque e impacto", "Perspectiva BSC", "Procesos internos"),
            ("5. Perfil de datos", "Fuente de origen", "dataset_pos_silver.parquet · columna es_anulacion"),
            ("5. Perfil de datos", "Frecuencia de reporte", "Mensual"),
            ("5. Perfil de datos", "Custodio del dato", "Analista de datos / administrador del POS"),
            ("6. Metas y rangos", "Meta objetivo", f"≤ {meta_anul:.1f}%"),
            ("6. Metas y rangos", "Semáforo (R/A/V)",
             f"Verde ≤ {meta_anul:.1f}% · Amarillo hasta {meta_anul * 1.5:.1f}% · Rojo > {meta_anul * 1.5:.1f}%"),
            ("7. Gobernanza", "Dueño del KPI (owner)", "Gerente de Operaciones"),
        ], columns=["Sección", "Campo técnico", "Especificación / contenido"])
        mostrar_tabla(ficha)

    por_t = por_mes_an.groupby("id_tienda")[["lineas", "anulaciones", "monto_anulado"]].sum()  # SUM por tienda
    por_t["tasa_pct"] = por_t["anulaciones"] / por_t["lineas"] * 100  # Tasa de anulación
    por_t["estado"] = por_t["tasa_pct"].apply(lambda v: estado_semaforo(v, meta_anul))  # Semáforo
    c1, c2 = st.columns([3, 2])
    with c1:
        colores = por_t["estado"].map(lambda e: "#2E8B57" if "🟢" in e else ("#E0A100" if "🟡" in e else "#B00020"))
        fig = go.Figure(go.Bar(x=por_t.index, y=por_t["tasa_pct"], marker_color=colores))
        fig.add_hline(y=meta_anul, line_dash="dash", annotation_text=f"Meta {meta_anul:.1f}%")
        fig.update_layout(title="Tasa de anulación por tienda (semáforo)", yaxis_title="% de líneas anuladas")
        st.plotly_chart(estilo(fig))
    with c2:
        mostrar_tabla(por_t.rename(columns={"lineas": "Líneas", "anulaciones": "Anulaciones",
                                            "monto_anulado": "Monto anulado", "tasa_pct": "Tasa %",
                                            "estado": "Estado"}).round(2))
    mm = por_mes_an.assign(tasa=por_mes_an["anulaciones"] / por_mes_an["lineas"] * 100)  # Tasa por tienda-mes
    hm = mm.pivot(index="id_tienda", columns="mes", values="tasa")   # Matriz tienda × mes
    hm.columns = [c.strftime("%Y-%m") for c in hm.columns]           # Etiquetas legibles de mes
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(color="Tasa %"),
                    title="Mapa de calor de la tasa de anulación (tienda × mes)")
    st.plotly_chart(estilo(fig, 360))
    st.markdown("**Diagnóstico por cajero (Top 10 con mayor tasa)**")
    min_l = st.number_input("Mínimo de líneas por cajero para ser evaluado", 50, 5000, 500, 50)  # Evita tasas inestables
    cj = por_cajero_an[por_cajero_an["lineas"] >= min_l].copy()      # Cajeros con volumen suficiente
    cj["tasa_pct"] = cj["anulaciones"] / cj["lineas"] * 100          # Tasa por cajero
    cj["estado"] = cj["tasa_pct"].apply(lambda v: estado_semaforo(v, meta_anul))  # Semáforo
    mostrar_tabla(cj.sort_values("tasa_pct", ascending=False).head(10).round(2))  # Top 10

# ===================== PESTAÑA 4 · AGREGACIÓN TEMPORAL Y CRECIMIENTO =====================
with tabs[3]:
    st.subheader("KPI 4 · Crecimiento Mensual del Ingreso Neto (%)")
    st.info(f"**Pregunta SMART:** ¿Cuál es la variación porcentual mensual del ingreso neto por tienda y cuáles "
            f"registran caídas superiores al {umbral_caida}% en meses completos, para reasignar las acciones "
            "comerciales y de dotación del siguiente mes?")
    ing_tm = (pre_f.groupby(["id_tienda", "mes"])
                   .agg(ingreso_neto=("ticket_neto", "sum"), precuentas=("id_precuenta", "count"),
                        ticket_prom=("ticket_neto", "mean")).reset_index())  # SUM, COUNT y AVG por tienda-mes
    piv_i = ing_tm.pivot(index="mes", columns="id_tienda", values="ingreso_neto").sort_index()  # mes × tienda
    mask_comp = completo.reindex(piv_i.index).fillna(False).astype(bool)  # Bandera de mes completo alineada
    piv_ok = piv_i.copy()
    piv_ok.loc[~mask_comp, :] = np.nan                               # Meses parciales no se comparan
    mom = ((piv_ok / piv_ok.shift(1) - 1) * 100).replace([np.inf, -np.inf], np.nan)  # Variación mensual %

    c1, c2 = st.columns([3, 2])
    with c1:
        tot_plot = piv_i.sum(axis=1)                                 # Ingreso total por mes (incluye parciales)
        colores = [COLOR_PRIMARIO if mask_comp.loc[m] else COLOR_NEUTRO for m in tot_plot.index]  # Gris = parcial
        fig = go.Figure(go.Bar(x=[m.strftime("%Y-%m") for m in tot_plot.index], y=tot_plot.values,
                               marker_color=colores))
        fig.update_layout(title="Ingreso neto mensual total (gris = mes parcial)", yaxis_title="Ingreso neto")
        st.plotly_chart(estilo(fig))
    with c2:
        d_tab = pd.DataFrame({"Días con datos": dias_datos, "Días del mes": dias_datos.index.days_in_month,
                              "Mes completo": completo})             # Control de completitud
        d_tab.index = d_tab.index.strftime("%Y-%m")
        mostrar_tabla(d_tab)

    hm = mom.T                                                       # tienda × mes
    hm.columns = [c.strftime("%Y-%m") for c in hm.columns]
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    labels=dict(color="MoM %"), title="Variación mensual del ingreso neto por tienda (%)")
    st.plotly_chart(estilo(fig, 360))

    caidas = mom.stack().reset_index()                               # Formato largo (se descartan NaN)
    caidas.columns = ["mes", "tienda", "variacion_pct"]
    caidas = caidas[caidas["variacion_pct"] < -umbral_caida].sort_values("variacion_pct")  # Solo caídas relevantes
    caidas["mes"] = caidas["mes"].dt.strftime("%Y-%m")
    st.markdown(f"**Alertas: tienda-mes con caída mayor al {umbral_caida}%** ({len(caidas)} casos)")
    mostrar_tabla(caidas.round(2))

    st.markdown("**Agregación por tienda (SUM · COUNT · AVG · GROUP BY)**")
    ag = pre_f.groupby("id_tienda").agg(ingreso_neto=("ticket_neto", "sum"), precuentas=("id_precuenta", "count"),
                                        ticket_prom=("ticket_neto", "mean"),
                                        tragos=("n_tragos", "sum"), piqueos=("n_piqueos", "sum")).reset_index()  # Agregaciones
    ag["tragos_por_cuenta"] = ag["tragos"] / ag["precuentas"]        # Ratios derivados (reemplazan a comensales)
    ag["piqueos_por_cuenta"] = ag["piqueos"] / ag["precuentas"]
    c3, c4 = st.columns(2)
    with c3:
        mostrar_tabla(ag.round(2))
    with c4:
        fig = px.bar(ag.sort_values("ingreso_neto"), x="ingreso_neto", y="id_tienda", orientation="h",
                     color_discrete_sequence=[COLOR_PRIMARIO], title="Ingreso neto acumulado por tienda")
        st.plotly_chart(estilo(fig))

# ===================== PESTAÑA 5 · TRAGOS VS PIQUEOS: CORRELACIÓN, ASOCIACIÓN Y CAUSALIDAD =====================
with tabs[4]:
    st.subheader("KPI 5 · Correlación (Spearman) entre Nº de Tragos y Nº de Piqueos por Precuenta")
    st.info("**Pregunta SMART:** ¿Qué intensidad tiene la correlación monótona entre el número de tragos y el "
            "número de piqueos por precuenta en el periodo filtrado, supera |ρ| = 0.50 (moderadamente fuerte) y "
            "ambos productos elevan el ticket neto, para justificar combos trago + piqueo en el próximo trimestre?")
    if not cats_trago or not cats_piq:
        st.warning("Selecciona en la barra lateral al menos una categoría para TRAGOS y una para PIQUEOS "
                   "(sección 'Variables de consumo').")
    else:
        st.caption(f"**Tragos** = {', '.join(cats_trago)} · **Piqueos** = {', '.join(cats_piq)}. "
                   "Cada variable es el número de unidades netas de esas categorías en la precuenta "
                   "(las anulaciones restan).")
        pvb = pre_f[pre_f["ticket_neto"] > 0]                        # Cuentas válidas del filtro
        q1c, q2c, q3c, q4c = st.columns(4)
        q1c.metric("Cuentas con tragos", f"{pvb['tiene_trago'].mean() * 100:.1f}%", f"{int(pvb['tiene_trago'].sum()):,}", delta_color="off")
        q2c.metric("Cuentas con piqueos", f"{pvb['tiene_piqueo'].mean() * 100:.1f}%", f"{int(pvb['tiene_piqueo'].sum()):,}", delta_color="off")
        q3c.metric("Cuentas con ambos", f"{(pvb['tiene_trago'] & pvb['tiene_piqueo']).mean() * 100:.1f}%",
                   f"{int((pvb['tiene_trago'] & pvb['tiene_piqueo']).sum()):,}", delta_color="off")
        q4c.metric("Máx. por cuenta (control de calidad)", f"{int(pvb['n_tragos'].max())} / {int(pvb['n_piqueos'].max())}",
                   "tragos / piqueos", delta_color="off")

        if "spearman_m" not in cr:
            if cr.get("sin_varianza"):
                st.warning("Tragos o piqueos no varían en el filtro (siempre el mismo valor): no se puede correlacionar. "
                           "Revisa las categorías elegidas.")
            else:
                st.warning("No hay suficientes precuentas válidas para calcular correlaciones.")
        else:
            st.markdown("#### 1 · ¿Hay correlación entre tragos y piqueos?")
            m1, m2, m3 = st.columns(3)
            m1.metric("Pearson (r) · lineal", f"{cr['pearson_r']:.3f}", f"{fuerza_correlacion(cr['pearson_r'])} · {fmt_p(cr['pearson_p'])}", delta_color="off")
            m2.metric("Spearman (ρ) · monótona", f"{cr['spearman_rho']:.3f}", f"{fuerza_correlacion(cr['spearman_rho'])} · {fmt_p(cr['spearman_p'])}", delta_color="off")
            m3.metric("Kendall (τ) · muestra 20 mil", f"{cr['kendall_tau']:.3f}", f"{fuerza_correlacion(cr['kendall_tau'])} · {fmt_p(cr['kendall_p'])}", delta_color="off")
            st.caption(f"Base: {cr['n']:,} precuentas ({base_corr.lower()}). Con tantos registros el p-valor será casi siempre ≈ 0: "
                       "la decisión debe apoyarse en la magnitud del coeficiente, no solo en la significancia.")
            dif = abs(cr["pearson_r"] - cr["spearman_rho"]) if not np.isnan(cr["pearson_r"] + cr["spearman_rho"]) else np.nan
            if not np.isnan(dif):
                if dif > 0.10:
                    st.warning(f"Pearson y Spearman difieren en {dif:.2f}: hay sensibilidad a valores atípicos o relación no lineal. "
                               f"Pearson sin valores extremos (≤ P99) = {cr['pearson_sin_outliers']:.3f}.")
                else:
                    st.success(f"Pearson y Spearman son coherentes (diferencia {dif:.2f}). "
                               f"Pearson sin valores extremos (≤ P99) = {cr['pearson_sin_outliers']:.3f}.")
            s1, s2 = st.columns(2)
            s1.metric("ρ Tragos – Ticket neto", f"{cr['rho_tragos_ticket']:.3f}", fuerza_correlacion(cr["rho_tragos_ticket"]), delta_color="off")
            s2.metric("ρ Piqueos – Ticket neto", f"{cr['rho_piqueos_ticket']:.3f}", fuerza_correlacion(cr["rho_piqueos_ticket"]), delta_color="off")

            etiq = {"n_tragos": "Tragos", "n_piqueos": "Piqueos", "n_lineas": "Líneas", "descuento": "Descuento",
                    "hora_dia": "Hora", "ticket_neto": "Ticket neto"}   # Nombres legibles para la matriz
            c1, c2 = st.columns(2)
            with c1:
                metodo = st.radio("Matriz de correlación", ["Pearson", "Spearman", "Kendall (muestra)"], horizontal=True)
                mat = {"Pearson": cr["pearson_m"], "Spearman": cr["spearman_m"], "Kendall (muestra)": cr["kendall_m"]}[metodo]
                fig_h, ax = plt.subplots(figsize=(6, 4.6))           # Figura de matplotlib
                sns.heatmap(mat.rename(index=etiq, columns=etiq), annot=True, fmt=".2f", cmap="RdBu_r",
                            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)  # Mapa de calor seaborn
                ax.set_title(f"Matriz de correlación · {metodo}")
                st.pyplot(fig_h)
                plt.close(fig_h)                                     # Libera memoria de la figura
            with c2:
                cz = cr["cruce"]                                     # Tabla cruzada tragos × piqueos (tope 10)
                fig = go.Figure(go.Heatmap(z=np.log10(1 + cz.values), x=[str(c) for c in cz.columns],
                                           y=[str(i) for i in cz.index], text=cz.values, texttemplate="%{text}",
                                           colorscale="Reds", showscale=False))
                fig.update_layout(title="Cuentas según Nº de tragos (filas) y Nº de piqueos (columnas); 10 = 10 o más",
                                  xaxis_title="Piqueos", yaxis_title="Tragos")
                st.plotly_chart(estilo(fig, 420))

            st.markdown("**Control de variables de confusión: Spearman tragos–piqueos dentro de cada tienda**")
            pt = cr["por_tienda"].reset_index()
            pt.columns = ["tienda", "rho"]
            fig = px.bar(pt, x="tienda", y="rho", color_discrete_sequence=[COLOR_PRIMARIO])
            fig.add_hline(y=0.5, line_dash="dash", annotation_text="ρ = 0.50")
            fig.update_layout(yaxis_range=[-1, 1], title="Estratificación por tienda: ¿la relación se mantiene?")
            st.plotly_chart(estilo(fig, 320))

        # ---------------- 2 · ASOCIACIÓN / CAUSALIDAD ----------------
        st.markdown("---")
        st.markdown("#### 2 · Si pido un trago, ¿también pido piqueos? (asociación y causalidad)")
        T, P = pvb["tiene_trago"], pvb["tiene_piqueo"]
        a_, b_, c_, d_ = int((T & P).sum()), int((T & ~P).sum()), int((~T & P).sum()), int((~T & ~P).sum())
        if min(a_ + b_, c_ + d_) == 0 or min(a_ + c_, b_ + d_) == 0:
            st.caption("No hay variación suficiente (faltan cuentas con/sin tragos o con/sin piqueos) para medir la asociación.")
        else:
            p_pt, p_pnt = a_ / (a_ + b_), c_ / (c_ + d_)             # P(piqueo | trago) y P(piqueo | sin trago)
            rr = p_pt / p_pnt if p_pnt > 0 else np.nan               # Riesgo relativo (lift)
            or_ = (a_ * d_) / (b_ * c_) if b_ * c_ > 0 else np.nan   # Razón de momios
            if min(a_, b_, c_, d_) > 0:                              # IC 95% del OR (Woolf)
                se_l = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
                or_lo, or_hi = np.exp(np.log(or_) - 1.96 * se_l), np.exp(np.log(or_) + 1.96 * se_l)
            else:
                or_lo = or_hi = np.nan
            chi2_a, p_a, _, _ = stats.chi2_contingency(np.array([[a_, b_], [c_, d_]]), correction=False)
            phi = np.sqrt(chi2_a / (a_ + b_ + c_ + d_)) * (1 if (np.isnan(or_) or or_ >= 1) else -1)  # Coeficiente phi

            x1, x2, x3, x4 = st.columns(4)
            x1.metric("P(piqueo | pidió trago)", f"{p_pt * 100:.1f}%", f"n = {a_ + b_:,}", delta_color="off")
            x2.metric("P(piqueo | no pidió trago)", f"{p_pnt * 100:.1f}%", f"n = {c_ + d_:,}", delta_color="off")
            x3.metric("Riesgo relativo (lift)", "n/d" if np.isnan(rr) else f"{rr:.2f}×", "piqueo con trago vs sin trago", delta_color="off")
            x4.metric("Razón de momios (OR)", "n/d" if np.isnan(or_) else f"{or_:.2f}",
                      "" if np.isnan(or_lo) else f"IC 95%: {or_lo:.2f} – {or_hi:.2f}", delta_color="off")
            st.markdown(f"χ²(1) = **{chi2_a:,.1f}**, {fmt_p(p_a)}, φ = **{phi:.3f}** ({fuerza_correlacion(phi)}).")

            # Control de confusión: OR de Mantel–Haenszel por tienda × tipo de día × franja horaria
            est = pvb.assign(celda=(T.astype(int) * 2 + P.astype(int)))
            cnt = (est.groupby(["id_tienda", "tipo_dia", "franja", "celda"]).size()
                      .unstack("celda").reindex(columns=[0, 1, 2, 3]).fillna(0))
            n_s = cnt.sum(axis=1)
            num = (cnt[3] * cnt[0] / n_s).sum()                      # Σ a·d / n
            den = (cnt[2] * cnt[1] / n_s).sum()                      # Σ b·c / n
            or_mh = num / den if den > 0 else np.nan
            st.markdown(f"**OR ajustado (Mantel–Haenszel por tienda × tipo de día × franja horaria) = "
                        f"{'n/d' if np.isnan(or_mh) else f'{or_mh:.2f}'}** frente al OR crudo de "
                        f"{'n/d' if np.isnan(or_) else f'{or_:.2f}'}.")
            if not (np.isnan(or_mh) or np.isnan(or_)):
                if abs(or_mh - or_) / or_ <= 0.20:
                    st.success("El OR casi no cambia al controlar tienda, tipo de día y horario: la asociación no se explica por esos factores.")
                else:
                    st.warning("El OR cambia más de 20% al controlar tienda, tipo de día y horario: parte de la asociación es "
                               "efecto de confusión (p. ej., en ciertas franjas se pide más de todo).")

            cp = (pvb.groupby(["id_tienda", "tiene_trago"])["tiene_piqueo"].mean().unstack() * 100)
            cp = cp.rename(columns={True: "Con trago", False: "Sin trago"}).reset_index()
            cols_ok = [c for c in ["Sin trago", "Con trago"] if c in cp.columns]
            fig = px.bar(cp.melt(id_vars="id_tienda", value_vars=cols_ok, var_name="Grupo", value_name="% con piqueo"),
                         x="id_tienda", y="% con piqueo", color="Grupo", barmode="group",
                         color_discrete_map={"Con trago": COLOR_PRIMARIO, "Sin trago": COLOR_NEUTRO},
                         labels={"id_tienda": "Tienda"}, title="% de cuentas con piqueo según si pidieron trago, por tienda")
            st.plotly_chart(estilo(fig, 330))

            # ---- Chi-cuadrado POR TIENDA: tragos vs piqueos (2×2 dentro de cada tienda) ----
            st.markdown("**Chi-cuadrado de independencia tragos × piqueos, calculado dentro de cada tienda**")
            filas_ch = []
            for tnd, gt in pvb.groupby("id_tienda"):
                Tt, Pt = gt["tiene_trago"], gt["tiene_piqueo"]
                a2, b2, c2, d2 = int((Tt & Pt).sum()), int((Tt & ~Pt).sum()), int((~Tt & Pt).sum()), int((~Tt & ~Pt).sum())
                n2 = a2 + b2 + c2 + d2
                if min(a2 + b2, c2 + d2) == 0 or min(a2 + c2, b2 + d2) == 0:   # Falta variación: no se puede calcular
                    continue
                obs2 = np.array([[a2, b2], [c2, d2]])
                chi_t, p_t, _, esp_t = stats.chi2_contingency(obs2, correction=False)
                or_t = (a2 * d2) / (b2 * c2) if b2 * c2 > 0 else np.nan
                phi_t = np.sqrt(chi_t / n2) * (1 if (np.isnan(or_t) or or_t >= 1) else -1)
                filas_ch.append({"Tienda": tnd, "Cuentas": n2,
                                 "% con trago": (a2 + b2) / n2 * 100, "% con piqueo": (a2 + c2) / n2 * 100,
                                 "% piqueo | con trago": a2 / (a2 + b2) * 100,
                                 "% piqueo | sin trago": c2 / (c2 + d2) * 100,
                                 "χ²(1)": chi_t, "p-valor": fmt_p(p_t), "φ": phi_t,
                                 "Fuerza": fuerza_correlacion(phi_t), "Razón de momios": or_t,
                                 "¿Asociación?": "Sí" if p_t <= alpha else "No",
                                 "Esperado ≥ 5": "Sí" if (esp_t >= 5).all() else "No"})
            if filas_ch:
                tabla_ch = pd.DataFrame(filas_ch)
                mostrar_tabla(tabla_ch.round(3))
                fig = px.bar(tabla_ch, x="Tienda", y="φ", color_discrete_sequence=[COLOR_PRIMARIO],
                             labels={"φ": "φ (asociación tragos–piqueos)"},
                             title="Intensidad de la asociación tragos–piqueos por tienda (φ)")
                fig.add_hline(y=0, line_color="black")
                st.plotly_chart(estilo(fig, 320))
                st.caption("φ es el equivalente a una correlación para dos variables de tipo sí/no (tiene trago, tiene piqueo): "
                           "0 = independientes, positivo = quien pide trago tiende a pedir piqueo. Con muchas cuentas el p-valor "
                           "casi siempre es significativo; decide con φ y el % de piqueo con y sin trago.")
            else:
                st.caption("Ninguna tienda tiene variación suficiente en tragos y piqueos para el chi-cuadrado.")

            seq = secuencia_pedido(tiendas_sel, f1, f2, cats_trago, cats_piq)
            st.markdown("**Secuencia temporal (¿qué se pide primero?)**")
            if seq:
                y1, y2, y3, y4 = st.columns(4)
                y1.metric("Cuentas con ambos", f"{seq['n']:,}", delta_color="off")
                y2.metric("Trago primero", f"{seq['trago_primero'] / seq['n'] * 100:.1f}%", delta_color="off")
                y3.metric("Piqueo primero", f"{seq['piqueo_primero'] / seq['n'] * 100:.1f}%", delta_color="off")
                y4.metric("Mismo instante", f"{seq['simultaneo'] / seq['n'] * 100:.1f}%", delta_color="off")
            else:
                st.caption("No disponible: falta la columna hora_pedido o no hay cuentas con ambos tipos de producto.")

            st.warning("**Asociación ≠ causalidad.** Los datos muestran si quien pide trago tiende a pedir piqueos, pero no "
                       "prueban que pedir el trago *cause* pedir el piqueo (puede ser el tipo de cliente, la hora o el plan "
                       "de la salida). Criterios para sostener causalidad: (1) secuencia temporal — evaluada arriba; "
                       "(2) mecanismo plausible — el maridaje trago/piqueo lo es, pero debe validarse con el equipo de sala; "
                       "(3) control de confusión — OR ajustado arriba; (4) experimento A/B — por ejemplo, que el mozo sugiera "
                       "un piqueo al tomar el pedido de tragos en una tienda piloto y comparar con una tienda control.")

        # ---------------- 3 · EFECTO EN EL TICKET ----------------
        st.markdown("---")
        st.markdown("#### 3 · ¿Tragos y piqueos elevan el ticket medio?")
        filas = []
        grupos = {}
        for perfil in PERFILES:
            s = pvb.loc[pvb["perfil"] == perfil, "ticket_neto"].to_numpy(float)
            if len(s) >= 2:
                m_, lo_, hi_ = ic_media(s, conf)
                filas.append({"Perfil de cuenta": perfil, "Cuentas": len(s), "% de cuentas": len(s) / len(pvb) * 100,
                              "Ticket medio": m_, f"IC {conf:.0%} inf.": lo_, f"IC {conf:.0%} sup.": hi_,
                              "Mediana": float(np.median(s))})
                grupos[perfil] = s
        if len(filas) >= 2:
            tp = pd.DataFrame(filas)
            mostrar_tabla(tp.round(2))
            fig = go.Figure(go.Bar(
                x=tp["Perfil de cuenta"], y=tp["Ticket medio"], marker_color=COLOR_PRIMARIO,
                error_y=dict(type="data", array=(tp[f"IC {conf:.0%} sup."] - tp["Ticket medio"]).to_numpy(),
                             arrayminus=(tp["Ticket medio"] - tp[f"IC {conf:.0%} inf."]).to_numpy())))
            fig.update_layout(title=f"Ticket neto medio según perfil de consumo (IC {conf:.0%})", yaxis_title="Ticket neto")
            st.plotly_chart(estilo(fig, 340))

            rng = np.random.default_rng(42)                          # Submuestras reproducibles para Kruskal-Wallis
            sub_g = [rng.choice(v, size=min(len(v), 100000), replace=False) for v in grupos.values()]
            kw_stat, kw_p = stats.kruskal(*sub_g)
            st.markdown(f"Kruskal–Wallis entre perfiles: H = **{kw_stat:,.1f}**, {fmt_p(kw_p)} → "
                        f"**{'los tickets difieren entre perfiles' if kw_p <= alpha else 'sin evidencia de diferencia'}** (α = {alpha}).")

            comp = []                                                # Comparaciones por pares con Welch + d de Cohen
            for g1, g2 in [("Tragos + piqueos", "Solo tragos"), ("Tragos + piqueos", "Solo piqueos"),
                           ("Solo tragos", "Sin tragos ni piqueos"), ("Solo piqueos", "Sin tragos ni piqueos")]:
                if g1 in grupos and g2 in grupos:
                    ga, gb = grupos[g1], grupos[g2]
                    t_, p_ = stats.ttest_ind(ga, gb, equal_var=False)
                    d_c = cohen_d(ga, gb)
                    comp.append({"Comparación": f"{g1} vs {g2}", "Diferencia de medias": ga.mean() - gb.mean(),
                                 "Brecha %": (ga.mean() / gb.mean() - 1) * 100, "t de Welch": t_,
                                 "p-valor": fmt_p(p_), "d de Cohen": d_c, "Efecto": etiqueta_efecto(d_c),
                                 "¿Significativa?": "Sí" if p_ <= alpha else "No"})
            if comp:
                mostrar_tabla(pd.DataFrame(comp).round(3))

        g_t = pvb.assign(t=pvb["n_tragos"].clip(upper=8),
                         p=pvb["n_piqueos"].clip(upper=3).map({0: "0 piqueos", 1: "1 piqueo", 2: "2 piqueos", 3: "3+ piqueos"}))
        lin = g_t.groupby(["t", "p"])["ticket_neto"].agg(["mean", "size"]).reset_index()
        lin = lin[lin["size"] >= 30]                                 # Solo celdas con volumen suficiente
        if not lin.empty:
            fig = px.line(lin, x="t", y="mean", color="p", markers=True,
                          category_orders={"p": ["0 piqueos", "1 piqueo", "2 piqueos", "3+ piqueos"]},
                          labels={"t": "Nº de tragos (8 = 8 o más)", "mean": "Ticket neto medio", "p": ""},
                          title="Ticket medio según Nº de tragos, separado por Nº de piqueos (celdas con ≥ 30 cuentas)")
            st.plotly_chart(estilo(fig, 380))

        mod = modelo_ticket(tiendas_sel, f1, f2, cats_trago, cats_piq)
        if mod:
            st.markdown("**Regresión lineal: ticket ~ tragos + piqueos + interacción**")
            mostrar_tabla(mod["tabla"].assign(p=mod["tabla"]["p"].map(fmt_p)).round(3))
            co = mod["tabla"].set_index("Término")["Coeficiente"]
            sinergia = ("positiva (el piqueo potencia el efecto del trago)" if co["Interacción trago × piqueo"] > 0
                        else "negativa (parte del gasto se solapa)")
            st.caption(f"R² = {mod['r2']:.3f} · n = {mod['n']:,} (ticket recortado en P99). Cada trago adicional se asocia con "
                       f"{co['Por cada trago']:+.2f} de ticket y cada piqueo adicional con {co['Por cada piqueo']:+.2f}, "
                       f"manteniendo el otro constante; la interacción es {sinergia}. Es un modelo asociativo, no causal.")

        st.markdown("**Escala estándar de interpretación (valor absoluto)**")
        mostrar_tabla(pd.DataFrame({
            "Rango |coeficiente|": ["0.00", "0.01 – 0.29", "0.30 – 0.49", "0.50 – 0.69", "0.70 – 0.99", "1.00"],
            "Fuerza": ["Nula", "Débil", "Moderada", "Moderadamente fuerte", "Fuerte / Alta", "Perfecta"],
            "Nota": ["Independencia", "Tendencia sutil", "Relación clara", "Patrón evidente",
                     "Una variable predice a la otra", "Rara en datos reales"]}))
        st.caption("Advertencias: el signo no resta fuerza; Kendall suele dar valores menores que Pearson/Spearman; "
                   "la exigencia del umbral depende de la disciplina.")

# ===================== PESTAÑA 6 · PRUEBAS DE HIPÓTESIS E INFERENCIA =====================
with tabs[5]:
    st.subheader("KPI 6 · Brecha Porcentual de Ticket Neto: Sábado y Domingo vs Lunes a Viernes")
    st.info(f"**Pregunta SMART:** ¿Difiere significativamente (α = {alpha}) el ticket neto promedio entre sábado–domingo "
            f"y lunes–viernes y cuál es la brecha porcentual con su IC al {conf:.0%}, para decidir el próximo "
            "trimestre una política de dotación y promociones diferenciada por tipo de día?")
    st.caption("H₀: μ(Sáb–Dom) = μ(Lun–Vie) · H₁: μ(Sáb–Dom) ≠ μ(Lun–Vie). Prueba bilateral por defecto.")
    if not res_b:
        st.warning("Se necesitan precuentas en ambos grupos de días para comparar.")
    else:
        r = res_b
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Ticket medio Sáb–Dom", f"{r['media_a']:,.2f}", f"n = {r['n_a']:,}", delta_color="off")
        a2.metric("Ticket medio Lun–Vie", f"{r['media_b']:,.2f}", f"n = {r['n_b']:,}", delta_color="off")
        a3.metric("Brecha", f"{r['brecha_pct']:+.2f}%", f"d de Cohen = {r['d']:.3f} ({etiqueta_efecto(r['d'])})", delta_color="off")
        a4.metric("t de Welch", f"{r['t']:.2f}", fmt_p(r["p"]), delta_color="off")
        sig = r["p"] <= alpha                                        # ¿Se rechaza H0?
        dire = "mayor" if r["dif"] > 0 else "menor"                  # Dirección del efecto
        st.markdown(
            f"**Redacción del informe:** {'Existe' if sig else 'No existe'} evidencia estadísticamente significativa "
            f"para rechazar H₀ (t({r['gl']:.0f}) = {r['t']:.2f}, {fmt_p(r['p'])}, α = {alpha}). "
            f"La diferencia de medias es {r['dif']:+.2f} (IC {conf:.0%}: {r['lo']:+.2f} a {r['hi']:+.2f}); el ticket de "
            f"sábado y domingo es {dire} que el de lunes a viernes. Mann-Whitney (robusto a atípicos): {fmt_p(r['p_u'])}.")
        if sig and etiqueta_efecto(r["d"]) in ("Trivial", "Pequeño"):
            st.warning("Paradoja de Big Data: la diferencia es estadísticamente significativa pero el efecto es "
                       f"{etiqueta_efecto(r['d']).lower()}. Evalúa su relevancia práctica antes de decidir.")
        fig = go.Figure()
        for nombre, (m_, lo_, hi_) in [("Sábado y Domingo", r["ic_a"]), ("Lunes a Viernes", r["ic_b"])]:
            fig.add_trace(go.Bar(x=[nombre], y=[m_], error_y=dict(type="data", array=[hi_ - m_], arrayminus=[m_ - lo_]),
                                 marker_color=COLOR_PRIMARIO if nombre.startswith("S") else COLOR_SECUNDARIO,
                                 name=nombre))
        fig.update_layout(title=f"Ticket neto medio con IC {conf:.0%}", yaxis_title="Ticket neto", showlegend=False)
        st.plotly_chart(estilo(fig, 340))

        # ---------------- Intervalo de confianza (fórmula z) ----------------
        st.markdown("---")
        st.markdown("### Intervalo de confianza")
        st.markdown("Rango de valores dentro del cual esperamos encontrar el parámetro real de la población, "
                    f"con un determinado nivel de confianza (1 − α) = **{conf:.0%}**.")
        st.latex(r"\bar{x} \pm z^{*} \cdot \frac{\sigma}{\sqrt{n}}")
        st.markdown("**Donde:**  \n"
                    "• **x̄**: promedio de la muestra  \n"
                    "• **z\\***: valor crítico que fija el margen de error (90% → 1.645 · 95% → 1.960 · 99% → 2.576)  \n"
                    "• **σ**: variación (desviación estándar)  \n"
                    "• **n**: tamaño de la muestra")
        z_star = float(stats.norm.ppf(1 - (1 - conf) / 2))           # Valor crítico z del nivel de confianza elegido
        filas_ic = []
        for nombre, mask in [("Sábado y Domingo", pv["tipo_dia"] == "Sábado y Domingo"),
                             ("Lunes a Viernes", pv["tipo_dia"] == "Lunes a Viernes")]:
            x_g = pv.loc[mask, "ticket_neto"].to_numpy(float)
            n_g, m_g, sd_g = len(x_g), x_g.mean(), x_g.std(ddof=1)   # n, x̄ y σ (desviación muestral)
            ee_g = sd_g / np.sqrt(n_g)                               # Error estándar σ/√n
            mg_g = z_star * ee_g                                     # Margen de error z*·σ/√n
            filas_ic.append({"Grupo": nombre, "n": n_g, "x̄ (promedio)": m_g, "σ": sd_g, "σ/√n": ee_g,
                             "z*": z_star, "Margen de error": mg_g,
                             "Límite inferior": m_g - mg_g, "Límite superior": m_g + mg_g})
        mostrar_tabla(pd.DataFrame(filas_ic).round(4))
        ic_a_txt = f"[{filas_ic[0]['Límite inferior']:,.3f} ; {filas_ic[0]['Límite superior']:,.3f}]"
        ic_b_txt = f"[{filas_ic[1]['Límite inferior']:,.3f} ; {filas_ic[1]['Límite superior']:,.3f}]"
        # IC de la diferencia (Welch) y de la brecha porcentual
        gb = r["media_b"]
        brecha_lo, brecha_hi = r["lo"] / gb * 100, r["hi"] / gb * 100  # Brecha % = diferencia / media Lun–Vie
        st.markdown(f"**Interpretación:** con {conf:.0%} de confianza, el ticket medio poblacional de sábado y domingo se "
                    f"encuentra en {ic_a_txt} y el de lunes a viernes en {ic_b_txt}. La diferencia de medias (Sáb–Dom − Lun–Vie) "
                    f"es {r['dif']:+.3f} con IC {conf:.0%} de {r['lo']:+.3f} a {r['hi']:+.3f}; expresada como brecha porcentual, "
                    f"{r['brecha_pct']:+.2f}% (IC {conf:.0%}: {brecha_lo:+.2f}% a {brecha_hi:+.2f}%).")
        if (r["lo"] > 0) or (r["hi"] < 0):
            st.success("El intervalo de la diferencia no contiene el 0: coherente con el rechazo de H₀.")
        else:
            st.info("El intervalo de la diferencia contiene el 0: coherente con no rechazar H₀.")
        st.caption("Con muestras tan grandes (n de cientos de miles) la distribución t y la normal coinciden, por eso se usa "
                   "z*. El intervalo de la diferencia usa el error estándar de Welch (no asume varianzas iguales) y la brecha "
                   "porcentual se obtiene dividiendo sus límites entre la media de Lun–Vie. Un intervalo muy estrecho refleja "
                   "el tamaño de muestra, no un efecto grande: revisa también el d de Cohen.")

    st.markdown("---")
    st.markdown("### Prueba t de una muestra: ticket vs valor de referencia")
    x1 = pv["ticket_neto"].values                                    # Ticket de todas las cuentas válidas
    ref = st.number_input("Valor de referencia (ticket objetivo)", value=float(round(x1.mean() * 1.05, 2)), step=0.1)
    alt = st.radio("Tipo de prueba", ["Bilateral (≠)", "Unilateral izquierda (<)", "Unilateral derecha (>)"], horizontal=True)
    alt_map = {"Bilateral (≠)": "two-sided", "Unilateral izquierda (<)": "less", "Unilateral derecha (>)": "greater"}
    t1 = stats.ttest_1samp(x1, ref, alternative=alt_map[alt])        # Prueba t contra el valor de referencia
    m_, lo_, hi_ = ic_media(x1, conf)                                # IC de la media
    st.markdown(f"Media observada = **{m_:,.3f}** (IC {conf:.0%}: {lo_:,.3f} – {hi_:,.3f}) · "
                f"t({len(x1) - 1:,}) = **{t1.statistic:.2f}**, {fmt_p(t1.pvalue)} → "
                f"**{'se rechaza H₀' if t1.pvalue <= alpha else 'no se rechaza H₀'}** con α = {alpha}.")

    st.markdown("### Prueba t pareada: mismo grupo de tiendas en dos momentos")
    piv_t = pv.pivot_table(index="id_tienda", columns="mes", values="ticket_neto", aggfunc="mean")  # Ticket medio tienda × mes
    meses_ok = [m for m in piv_t.columns if bool(completo.get(m, False))]  # Solo meses completos
    if len(meses_ok) >= 2 and len(piv_t) >= 2:
        mA, mB = meses_ok[0], meses_ok[-1]                           # Primer y último mes completo
        par = piv_t[[mA, mB]].dropna()                               # Tiendas con dato en ambos momentos
        if len(par) >= 2:
            tp_ = stats.ttest_rel(par[mB], par[mA])                  # t pareada: después − antes
            st.markdown(f"Ticket medio por tienda, **{mA:%Y-%m}** → **{mB:%Y-%m}**: "
                        f"diferencia media = {(par[mB] - par[mA]).mean():+.3f} · t({len(par) - 1}) = {tp_.statistic:.2f}, "
                        f"{fmt_p(tp_.pvalue)} → **{'se rechaza H₀' if tp_.pvalue <= alpha else 'no se rechaza H₀'}** (n = {len(par)} tiendas).")
        else:
            st.caption("Menos de 2 tiendas con datos en ambos meses.")
    else:
        st.caption("Se requieren al menos 2 meses completos y 2 tiendas en el filtro.")

    st.markdown("### Chi-cuadrado de independencia con categorías de tragos y piqueos")
    st.caption("La prueba trabaja con CONTEOS de cuentas en cada combinación de categorías; los porcentajes solo se usan "
               "para leer y graficar. H₀: las dos variables son independientes · H₁: están asociadas.")
    if not cats_trago or not cats_piq:
        st.warning("Selecciona categorías de TRAGOS y PIQUEOS en la barra lateral (sección 'Variables de consumo').")
    else:
        NIV_T = ["0", "1-2", "3-4", "5+"]                            # Niveles de tragos por cuenta
        NIV_P = ["0", "1", "2", "3+"]                                # Niveles de piqueos por cuenta
        dfc = pre_f.copy()
        dfc["nivel_tragos"] = pd.cut(dfc["n_tragos"], [-1, 0, 2, 4, np.inf], labels=NIV_T).astype(str)
        dfc["nivel_piqueos"] = pd.cut(dfc["n_piqueos"], [-1, 0, 1, 2, np.inf], labels=NIV_P).astype(str)
        VARS = {"Nivel de tragos": ("nivel_tragos", NIV_T), "Nivel de piqueos": ("nivel_piqueos", NIV_P),
                "Perfil de cuenta": ("perfil", PERFILES), "Tienda": ("id_tienda", None),
                "Tipo de día": ("tipo_dia", None),
                "Franja horaria": ("franja", ["Madrugada (0–5h)", "Mañana (6–11h)", "Tarde (12–17h)",
                                              "Noche (18–21h)", "Noche tardía (22–23h)"])}
        cq1, cq2 = st.columns(2)
        with cq1:
            nom_f = st.selectbox("Variable en filas", list(VARS), index=0, key="chi_filas")
        with cq2:
            opciones_c = [v for v in VARS if v != nom_f]
            nom_c = st.selectbox("Variable en columnas", opciones_c,
                                 index=opciones_c.index("Tienda") if "Tienda" in opciones_c else 0, key="chi_cols")
        (col_f, ord_f), (col_c, ord_c) = VARS[nom_f], VARS[nom_c]
        ctab = pd.crosstab(dfc[col_f], dfc[col_c])                   # Tabla de CONTEOS observados
        if ord_f: ctab = ctab.reindex([x for x in ord_f if x in ctab.index])
        if ord_c: ctab = ctab[[x for x in ord_c if x in ctab.columns]]
        if ctab.shape[0] >= 2 and ctab.shape[1] >= 2:
            chi2, p_chi, gl_chi, esp = stats.chi2_contingency(ctab)  # χ², p, gl y frecuencias esperadas
            n_t = ctab.values.sum()
            v_cr = np.sqrt(chi2 / (n_t * (min(ctab.shape) - 1)))     # V de Cramér
            ok5 = (esp >= 5).mean() * 100                            # Supuesto: esperado ≥ 5 en las celdas
            st.markdown(f"**{nom_f} × {nom_c}**: χ²({gl_chi}) = **{chi2:,.1f}**, {fmt_p(p_chi)}, "
                        f"V de Cramér = **{v_cr:.3f}** ({fuerza_correlacion(v_cr)}) → "
                        f"**{'se rechaza independencia: hay asociación' if p_chi <= alpha else 'no se rechaza independencia'}** "
                        f"(α = {alpha}). Celdas con esperado ≥ 5: {ok5:.0f}%.")
            if ok5 < 80:
                st.warning("Menos del 80% de las celdas tiene frecuencia esperada ≥ 5: la prueba pierde validez. "
                           "Agrupa categorías con pocas cuentas (por ejemplo, junta '5+' con '3-4').")
            if p_chi <= alpha and v_cr < 0.10:
                st.warning("Significativa pero con efecto muy pequeño (V < 0.10): con muchas cuentas casi cualquier "
                           "diferencia sale significativa.")
            pct = ctab.div(ctab.sum(axis=1), axis=0) * 100           # % por fila (solo para leer/graficar)
            larga = pct.reset_index().melt(id_vars=col_f, var_name=nom_c, value_name="pct")
            fig = px.bar(larga, x=col_f, y="pct", color=nom_c, barmode="stack",
                         labels={col_f: nom_f, "pct": f"% de cuentas (dentro de cada {nom_f.lower()})"},
                         title=f"Composición de {nom_c.lower()} según {nom_f.lower()} (100% por barra)")
            st.plotly_chart(estilo(fig, 380))
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Conteos observados (cuentas)**")
                mostrar_tabla(ctab)
            with r2:
                resid = (ctab - esp) / np.sqrt(esp)                  # Residuos estandarizados
                st.markdown("**Residuos estandarizados** (|valor| > 2 = celda que se aparta de lo esperado)")
                fig_r = px.imshow(resid.round(1), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                                  color_continuous_midpoint=0, labels=dict(color="Residuo"))
                st.plotly_chart(estilo(fig_r, 320))
        else:
            st.caption("Se necesitan al menos 2 categorías en cada variable con cuentas en el filtro actual.")

    st.markdown("### Chi-cuadrado de bondad de ajuste: ¿las cuentas se reparten según los días disponibles?")
    obs = pre_f.groupby("dia_num").size().reindex(range(7), fill_value=0)  # Cuentas observadas por día
    dias_dato = pre_f.groupby("dia_num")["fecha_transaccion"].nunique().reindex(range(7), fill_value=0)  # Días con datos
    if (dias_dato > 0).all():                                        # Requiere los 7 días presentes
        esp_d = dias_dato / dias_dato.sum() * obs.sum()              # Esperado: proporcional a los días disponibles
        chi_g, p_g = stats.chisquare(obs.values, f_exp=esp_d.values)  # Prueba de bondad de ajuste
        w_ef = np.sqrt(chi_g / obs.sum())                            # Tamaño del efecto w de Cohen
        st.markdown(f"χ²(6) = **{chi_g:,.1f}**, {fmt_p(p_g)}, w = **{w_ef:.3f}** → "
                    f"**{'la distribución difiere de la esperada' if p_g <= alpha else 'sin evidencia de diferencia'}**.")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=DIAS_NOMBRE, y=obs.values, name="Observado", marker_color=COLOR_PRIMARIO))
        fig.add_trace(go.Bar(x=DIAS_NOMBRE, y=esp_d.values, name="Esperado", marker_color=COLOR_SECUNDARIO))
        fig.update_layout(barmode="group", title="Precuentas por día: observado vs esperado")
        st.plotly_chart(estilo(fig, 320))
    else:
        st.caption("El filtro no contiene los siete días de la jornada; amplía el rango de fechas.")

    st.markdown("---")
    st.caption("Interpretación correcta del IC: si repitiéramos el muestreo muchas veces, el "
               f"{conf:.0%} de los intervalos construidos contendría el parámetro verdadero; el parámetro es fijo, "
               "el intervalo es el que varía. Con muestras masivas, reporta siempre tamaño del efecto e IC.")

