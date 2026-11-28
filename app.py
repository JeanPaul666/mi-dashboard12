# -*- coding: utf-8 -*-
"""
Dashboard de ventas Rústica – esquema de 3 capas + accesibilidad (WCAG 2.1 / Okabe-Ito)
CAPA 1: Resumen ejecutivo   (¿Cómo vamos?)
CAPA 2: Tendencias          (¿Dónde está ocurriendo?)  -> vista global: evolución, composición,
                                                          comparativa, descomposición de la brecha, días
CAPA 3: Diagnóstico         (¿Por qué pasa esto?)      -> detalle: top familias, top productos y tablas
Dataset: dataset_pos_silver.parquet (en la raíz del repositorio)

Criterios de accesibilidad aplicados (Sesión 12):
  * Paleta Okabe-Ito (azul = cumple / naranja = no cumple), nada de rojo-verde.
  * Nunca solo color (WCAG 1.4.1): etiquetas, flechas ▲▼, texto de estado y tramas (rayado).
  * Contraste de texto >= 7:1 (AAA) y de gráficos >= 3:1 (WCAG 1.4.11). Verificado con WebAIM.
  * Simulador de visión (protanopia, deuteranopia, escala de grises) y modo Alto Contraste AAA.
  * Texto alternativo en cada gráfico, títulos descriptivos en paneles y ejes, chunking por capas.
"""
import calendar
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

st.set_page_config(page_title="Dashboard de Ventas – Rústica", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "dataset_pos_silver.parquet"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
T1, T2 = "Turno 1 (10:00–19:00)", "Turno 2 (19:00–03:00)"
WEBAIM = "https://webaim.org/resources/contrastchecker/"

# ----------------------------------------------------------------------------
# Paletas accesibles (Okabe-Ito) + simulador de visión
# ----------------------------------------------------------------------------
# Okabe-Ito oficiales: azul #0072B2, naranja #E69F00, celeste #56B4E9, bermellón #D55E00, púrpura #CC79A7
PALETAS = {
    "normal": dict(
        BG="#FFFFFF", TXT="#1A1A1A", TXT2="#404040", CARD="#F4F6F8", GRID="#D9D9D9",
        OK="#0072B2",        # cumple meta  (azul Okabe-Ito)
        NO="#E69F00",        # no cumple    (naranja Okabe-Ito) -> siempre con trama + borde oscuro
        NEUTRO="#767676", META="#1A1A1A", MOVIL="#D55E00", PREV="#CC79A7",
        TAG1="#7B1426", TAG2="#004B78", TAG3="#1A1A1A", TAGTXT="#FFFFFF",
        PILL_OK=("#E3F1FA", "#00395C"), PILL_NO=("#FFF1D6", "#6B3F00"), PILL_NEU=("#EDEDED", "#1A1A1A"),
        BW=1.2, LW=1.0,
    ),
    "alto": dict(  # Alto contraste AAA: negro sobre blanco, bordes gruesos
        BG="#FFFFFF", TXT="#000000", TXT2="#000000", CARD="#FFFFFF", GRID="#BDBDBD",
        OK="#00507A", NO="#E69F00", NEUTRO="#595959", META="#000000", MOVIL="#A34700", PREV="#8A3B6B",
        TAG1="#000000", TAG2="#000000", TAG3="#000000", TAGTXT="#FFFFFF",
        PILL_OK=("#FFFFFF", "#00304F"), PILL_NO=("#FFFFFF", "#4A2A00"), PILL_NEU=("#FFFFFF", "#000000"),
        BW=2.2, LW=2.0,
    ),
}

# Matrices de Machado et al. (2009), severidad 1.0, aplicadas en RGB lineal
_M_PROT = np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]])
_M_DEUT = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])


def _a_lineal(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _a_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simular_hex(h, modo):
    """Devuelve el color tal como lo vería una persona con la condición elegida."""
    if modo == "Normal" or not isinstance(h, str) or not h.startswith("#") or len(h) != 7:
        return h
    rgb = _a_lineal(np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]))
    if modo == "Escala de grises":
        y = float(rgb @ np.array([0.2126, 0.7152, 0.0722]))
        out = np.array([y, y, y])
    else:
        out = (_M_PROT if modo == "Protanopia" else _M_DEUT) @ rgb
    out = (_a_srgb(out) * 255).round().astype(int)
    return "#{:02X}{:02X}{:02X}".format(*out)


def lum_rel(h):
    r, g, b = _a_lineal(np.array([int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contraste(a, b):
    """Razón de contraste WCAG (misma fórmula que WebAIM Contrast Checker)."""
    la, lb = sorted([lum_rel(a), lum_rel(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


# ---- Barra superior: simulador de visión + alto contraste (como en el modelo del profesor) ----
with st.container(border=True):
    c_sim, c_hc = st.columns([3.4, 1])
    with c_sim:
        modo_vis = st.radio("👁 Simulador de visión", ["Normal", "Protanopia", "Deuteranopia", "Escala de grises"],
                            horizontal=True, help="Muestra los colores del dashboard como los ve una persona con "
                                                  "daltonismo. Si el mensaje se sigue entendiendo, el diseño es accesible.")
    with c_hc:
        alto = st.checkbox("◐ Alto contraste AAA", help="Texto negro sobre blanco, bordes gruesos y colores más oscuros.")

BASE = PALETAS["alto" if alto else "normal"]          # colores reales (para auditar contraste)
P = {k: (tuple(simular_hex(x, modo_vis) for x in v) if isinstance(v, tuple) else
         (simular_hex(v, modo_vis) if isinstance(v, str) else v)) for k, v in BASE.items()}
# El texto y los fondos neutros nunca cambian de tono; solo los colores de datos/estado se simulan.
for k in ("BG", "TXT", "TXT2", "CARD", "GRID"):
    P[k] = BASE[k]

CSS = """
<style>
.block-container {padding-top: 1.2rem;}
.stApp {background:__BG__; color:__TXT__; __FILTRO__}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label, .stApp .stMarkdown {color:__TXT__;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:__TXT2__ !important; opacity:1 !important; font-size:.88rem;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {color:__TXT2__ !important; opacity:1 !important;}
*:focus-visible {outline:3px solid __TXT__ !important; outline-offset:2px;}
.capa {display:flex; align-items:center; flex-wrap:wrap; gap:12px; margin: 6px 0 10px 0;}
.capa .tag {color:__TAGTXT__; font-weight:800; padding:6px 28px 6px 16px;
            clip-path: polygon(0 0, 88% 0, 100% 50%, 88% 100%, 0 100%); background:__TAG1__;}
.capa .tag.c2 {background:__TAG2__;} .capa .tag.c3 {background:__TAG3__;}
.capa .tit {font-weight:800; font-size:1.08rem; color:__TXT__;}
.capa .preg {color:__TXT__; text-decoration:underline; text-decoration-thickness:2px; text-underline-offset:3px;}
.capa .sub {color:__TXT2__; font-size:.88rem;}
.kpi {border:__BW__px solid __TXT2__; background:__CARD__; color:__TXT__; border-radius:10px; padding:14px 16px; height:100%;}
.kpi .lbl {font-size:.78rem; font-weight:800; letter-spacing:.04em; color:__TXT2__;}
.kpi .val {font-size:2rem; font-weight:800; line-height:1.15; color:__TXT__;}
.kpi .sub {font-size:.85rem; color:__TXT2__;}
.pill {display:inline-block; margin-top:6px; padding:3px 10px; border-radius:6px; font-size:.86rem; font-weight:800;}
.pill.mal {background:__PNO_BG__; color:__PNO_TX__; border:__BW__px solid __PNO_TX__;}
.pill.bien {background:__POK_BG__; color:__POK_TX__; border:__BW__px solid __POK_TX__;}
.pill.alerta {background:__PNE_BG__; color:__PNE_TX__; border:__BW__px solid __PNE_TX__;}
.bar {height:12px; border-radius:6px; background:__BG__; border:__BW__px solid __TXT__; margin:8px 0 4px 0; overflow:hidden;}
.bar > div {height:100%;}
.bar > div.ok {background:__OK__;}
.bar > div.no {background:repeating-linear-gradient(45deg, __NO__ 0, __NO__ 5px, __BG__ 5px, __BG__ 8px);}
.concl {border-radius:8px; padding:10px 14px; background:__PNO_BG__; color:__PNO_TX__; border:__BW__px solid __PNO_TX__;
        border-left-width:8px; font-size:.95rem;}
.concl.ok {background:__POK_BG__; color:__POK_TX__; border-color:__POK_TX__;}
.alt {font-size:.85rem; color:__TXT2__; border-left:4px solid __TXT2__; padding:2px 10px; margin:4px 0 10px 0;}
.alt b {color:__TXT__;}
</style>
"""
_tok = {
    "__BG__": P["BG"], "__TXT__": P["TXT"], "__TXT2__": P["TXT2"], "__CARD__": P["CARD"], "__OK__": P["OK"], "__NO__": P["NO"],
    "__TAG1__": P["TAG1"], "__TAG2__": P["TAG2"], "__TAG3__": P["TAG3"], "__TAGTXT__": P["TAGTXT"],
    "__PNO_BG__": P["PILL_NO"][0], "__PNO_TX__": P["PILL_NO"][1], "__POK_BG__": P["PILL_OK"][0],
    "__POK_TX__": P["PILL_OK"][1], "__PNE_BG__": P["PILL_NEU"][0], "__PNE_TX__": P["PILL_NEU"][1],
    "__BW__": f"{P['BW']:g}", "__FILTRO__": "filter:grayscale(1);" if modo_vis == "Escala de grises" else "",
}
for _k, _v in _tok.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def esc(x):
    return html.escape(str(x))


def soles(x):
    """Formatea montos: S/ 570 mil, S/ 1.2 M."""
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1e6:
        return f"S/ {x / 1e6:,.2f} M"
    if abs(x) >= 1e3:
        return f"S/ {x / 1e3:,.1f} mil"
    return f"S/ {x:,.0f}"


def soles_s(x):
    """Monto con signo explícito (+/−) para no depender del color."""
    return f"{'+' if x >= 0 else '−'}{soles(abs(x))}"


def var_pct(a, b):
    """Variación porcentual de a respecto a b (None si no se puede calcular)."""
    return None if (b is None or pd.isna(b) or b == 0) else (a / b - 1) * 100


def lbl_ym(ym):
    return f"{MESES[ym % 100 - 1]} {ym // 100}"


def mostrar(fig):
    """Estilo accesible común a todos los gráficos + compatibilidad entre versiones de Streamlit."""
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.14, font=dict(color=P["TXT"])),
        paper_bgcolor=P["BG"], plot_bgcolor=P["BG"], font=dict(color=P["TXT"], size=13),
        hoverlabel=dict(bgcolor="#FFFFFF", font=dict(color="#1A1A1A", size=13)),
    )
    fig.update_xaxes(linecolor=P["TXT"], gridcolor=P["GRID"], tickfont=dict(color=P["TXT"]), title_font=dict(color=P["TXT"]),
                     automargin=True, title_standoff=8)
    fig.update_yaxes(linecolor=P["TXT"], gridcolor=P["GRID"], tickfont=dict(color=P["TXT"]), title_font=dict(color=P["TXT"]), automargin=True)
    try:
        st.plotly_chart(fig, width="stretch", theme=None)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None)


def tabla(df, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def alt(texto):
    """Texto alternativo visible: la versión 'hablada' del gráfico (WCAG 1.1.1)."""
    st.markdown(f'<div class="alt" role="note"><b>Descripción del gráfico:</b> {esc(texto)}</div>', unsafe_allow_html=True)


def cabecera(n, titulo, pregunta, sub):
    st.markdown(
        f'<div class="capa" role="heading" aria-level="2"><span class="tag c{n}">CAPA {n}</span>'
        f'<span class="tit">{titulo} <span class="preg">{pregunta}</span></span>'
        f'<span class="sub">{sub}</span></div>',
        unsafe_allow_html=True,
    )


def tarjeta(titulo, valor, sub="", pill="", clase="", barra=None, estado_barra="ok"):
    barra_html = ""
    if barra is not None:
        w = min(max(barra, 0), 100)
        barra_html = (f'<div class="bar" role="progressbar" aria-label="{esc(titulo)}" aria-valuemin="0" '
                      f'aria-valuemax="100" aria-valuenow="{w:.0f}"><div class="{estado_barra}" style="width:{w:.2f}%"></div></div>')
    pill_html = f'<div class="pill {clase}">{pill}</div>' if pill else ""
    st.markdown(
        f'<div class="kpi" role="group" aria-label="{esc(titulo)}"><div class="lbl">{titulo}</div><div class="val">{valor}</div>'
        f'{barra_html}<div class="sub">{sub}</div>{pill_html}</div>',
        unsafe_allow_html=True,
    )


def marcador_ok(**extra):
    return dict(color=P["OK"], line=dict(color=P["TXT"], width=P["BW"]), **extra)


def marcador_no(**extra):
    """'No cumple' = naranja + RAYADO + borde oscuro: se distingue aunque se pierda el color."""
    return dict(color=P["NO"], line=dict(color=P["TXT"], width=P["BW"]),
                pattern=dict(shape="/", fgcolor=P["TXT"], bgcolor=P["NO"], size=7, solidity=0.35, fillmode="replace"), **extra)


# ----------------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------------
origen = None
if DATA_FILE.exists():
    with open(DATA_FILE, "rb") as fh:
        if fh.read(20).startswith(b"version https://git"):
            st.error(
                "El archivo .parquet es solo un puntero de Git LFS (no tiene los datos). "
                "Sube el parquet sin LFS (si pesa menos de 100 MB) o súbelo aquí abajo."
            )
        else:
            origen = str(DATA_FILE)
if origen is None:
    up = st.file_uploader("Sube dataset_pos_silver.parquet", type="parquet")
    if up is None:
        st.stop()
    origen = up.getvalue()

esquema = pq.read_schema(io.BytesIO(origen) if isinstance(origen, bytes) else origen).names

# ----------------------------------------------------------------------------
# Sidebar: mapeo de columnas
# ----------------------------------------------------------------------------
CAMPOS = {  # clave: (etiqueta, palabras para autodetectar, obligatorio)
    "fecha": ("Fecha", ["fecha", "date"], True),
    "venta": ("Venta (S/)", ["venta", "importe", "monto", "total", "subtotal"], True),
    "trans": ("Nº de comanda / precuenta", ["precuenta", "comanda", "pedido", "ticket", "documento", "nro", "id"], False),
    "familia": ("Familia (código o nombre)", ["familia", "categoria", "grupo"], False),
    "familia_nombre": ("Familia (nombre, si existe aparte)", ["categoria_familia", "nombre_familia", "familia_nombre", "desc_familia", "familia_desc", "nom_familia", "familia_nom", "descripcion_familia"], False),
    "producto": ("Producto", ["producto", "plato", "articulo", "item", "descripcion"], False),
    "local": ("Local", ["local", "sucursal", "tienda"], False),
    "hora": ("Hora (opcional)", ["hora"], False),
    "comens": ("Nº de comensales (opcional)", ["comensal", "pax", "persona"], False),
}


def adivinar(claves):
    for k in claves:
        for c in esquema:
            if k in c.lower():
                return c
    return None


def adivinar_familia_codigo():
    """Columna de familia que NO sea la de nombre (evita confundirlas)."""
    nom = adivinar(CAMPOS["familia_nombre"][1])
    for c in esquema:  # código de familia: contiene "famil", no es el nombre ni la subfamilia
        cl = c.lower()
        if "famil" in cl and "sub" not in cl and c != nom:
            return c
    return None


st.sidebar.title("📊 Filtros")
mapa = {}
with st.sidebar.expander("⚙️ Columnas del dataset", expanded=False):
    st.caption("Se detectan automáticamente. Corrígelas si alguna no corresponde.")
    for k, (etq, claves, obl) in CAMPOS.items():
        opciones = ["(ninguna)"] + esquema
        g = adivinar_familia_codigo() if k == "familia" else adivinar(claves)
        if k == "familia" and g is None:
            g = adivinar(["categoria", "grupo"])
        sel = st.selectbox(etq, opciones, index=opciones.index(g) if g else 0, key=f"map_{k}")
        mapa[k] = None if sel == "(ninguna)" else sel

if not mapa["fecha"] or not mapa["venta"]:
    st.error("Indica al menos las columnas de **Fecha** y **Venta** en el panel lateral.")
    st.stop()


@st.cache_data(show_spinner="Leyendo y preparando datos…")
def preparar(origen, mapa_t):
    m = dict(mapa_t)
    cols = sorted({v for v in m.values() if v})
    raw = pd.read_parquet(io.BytesIO(origen) if isinstance(origen, bytes) else origen, columns=cols)
    out = pd.DataFrame(index=raw.index)

    f = pd.to_datetime(raw[m["fecha"]], errors="coerce")
    out["fecha_dia"] = f.dt.normalize()
    out["venta"] = pd.to_numeric(raw[m["venta"]], errors="coerce").fillna(0.0)
    out["trans"] = raw[m["trans"]] if m["trans"] else np.arange(len(raw))

    # Familia: se prefiere la columna con el NOMBRE; si solo hay código, se muestra "Familia 49"
    col_fam = m.get("familia_nombre") or m["familia"]
    if col_fam:
        fam = raw[col_fam]
        if m.get("familia_nombre") is None:
            num = pd.to_numeric(fam, errors="coerce")
            if num.notna().mean() > 0.9:  # son códigos numéricos
                fam = num.round().astype("Int64").astype("string").radd("Familia ")
        out["familia"] = fam.astype("string").fillna("Sin dato").astype("category")
    else:
        out["familia"] = pd.Series("Sin dato", index=out.index, dtype="category")

    for k in ["producto", "local"]:
        if m[k]:
            out[k] = raw[m[k]].astype("string").fillna("Sin dato").astype("category")
        else:
            out[k] = pd.Series("Sin dato", index=out.index, dtype="category")

    # Hora -> turno (turno 1: 10-19 h, turno 2: 19-03 h)
    h = None
    if m["hora"]:
        s = raw[m["hora"]]
        if pd.api.types.is_datetime64_any_dtype(s):
            h = s.dt.hour
        else:
            h = pd.to_numeric(s, errors="coerce")
            if h.isna().mean() > 0.5:
                h = pd.to_datetime(s.astype(str), errors="coerce").dt.hour
    if h is None or h.isna().all():
        h = f.dt.hour
    out["hora"] = h
    out["turno"] = pd.Categorical(
        np.select([(h >= 10) & (h < 19), (h >= 19) | (h <= 3)], [T1, T2], default="Fuera de turno")
    )
    out["comens"] = pd.to_numeric(raw[m["comens"]], errors="coerce") if m["comens"] else np.nan

    out = out.dropna(subset=["fecha_dia"])
    out["ym"] = (out.fecha_dia.dt.year * 100 + out.fecha_dia.dt.month).astype("int32")
    return out


df = preparar(origen, tuple(sorted(mapa.items(), key=lambda x: x[0])))
if df.empty:
    st.error("No se pudieron interpretar las fechas. Revisa la columna de Fecha.")
    st.stop()
tiene_turno = df["hora"].nunique() > 1

# ----------------------------------------------------------------------------
# Sidebar: filtros, meta y ticket
# ----------------------------------------------------------------------------
yms = sorted(df["ym"].unique())
por_defecto = yms.index(202501) if 202501 in yms else len(yms) - 1
ym = st.sidebar.selectbox("Mes de análisis", yms, index=por_defecto, format_func=lbl_ym)
ym_prev = ym - 100
st.sidebar.caption(f"Comparativa: {lbl_ym(ym_prev)}")

locales = sorted(df["local"].cat.categories.astype(str))
f_local = st.sidebar.multiselect("Local", locales, default=locales) if len(locales) > 1 else locales

st.sidebar.markdown("**Meta de venta del mes**")
crec = st.sidebar.slider("Crecimiento objetivo vs año anterior (%)", -20, 50, 5)
meta_manual = st.sidebar.checkbox("Ingresar meta manual (S/)")
meta_valor = st.sidebar.number_input("Meta (S/)", min_value=0.0, value=500000.0, step=10000.0) if meta_manual else None

st.sidebar.markdown("**Ticket promedio**")
opciones_ticket = ["Por transacción"] + (["Por comensal (depurado)"] if mapa["comens"] else [])
modo_ticket = st.sidebar.radio("Cálculo", opciones_ticket, help="El nº de comensales tiene problemas de calidad; "
                               "al depurar se excluyen comandas fuera del rango permitido.")
tope_comens = st.sidebar.number_input("Máx. comensales por comanda", 1, 1000, 30) if modo_ticket != "Por transacción" else 30

st.sidebar.markdown("**Accesibilidad**")
st.sidebar.markdown(f"[Verificador de contraste WebAIM ↗]({WEBAIM}) — la auditoría de este dashboard está al final de la página.")

# ----------------------------------------------------------------------------
# Subconjuntos y KPIs
# ----------------------------------------------------------------------------
base = df[df["ym"].isin([ym, ym_prev])]
base = base[base["local"].astype(str).isin(f_local)].copy()
base["tx_key"] = (base["fecha_dia"].astype(str) + "|" + base["local"].astype(str) + "|" + base["trans"].astype(str))
cur, prev = base[base["ym"] == ym], base[base["ym"] == ym_prev]


def kpis(d):
    if d.empty:
        return dict(venta=0.0, trans=0, ticket=np.nan)
    agg = {"venta": ("venta", "sum")}
    if d["comens"].notna().any():
        agg["comens"] = ("comens", "max")
    tx = d.groupby("tx_key", sort=False).agg(**agg)
    venta, n = float(tx["venta"].sum()), len(tx)
    if modo_ticket != "Por transacción" and "comens" in tx:
        ok = tx["comens"].between(1, tope_comens)
        ticket = tx.loc[ok, "venta"].sum() / tx.loc[ok, "comens"].sum() if ok.any() else np.nan
    else:
        ticket = venta / n if n else np.nan
    return dict(venta=venta, trans=n, ticket=ticket)


k, kp = kpis(cur), kpis(prev)
if cur.empty:
    st.warning("No hay datos para el mes y filtros seleccionados.")
    st.stop()

if meta_manual:
    meta = meta_valor
elif kp["venta"] > 0:
    meta = kp["venta"] * (1 + crec / 100)
else:
    st.sidebar.warning(f"No hay ventas de {lbl_ym(ym_prev)} para calcular la meta automática. Ingresa la meta manual.")
    meta = st.sidebar.number_input("Meta del mes (S/)", min_value=1.0, value=500000.0, step=10000.0, key="meta_fallback")
cumpl = k["venta"] / meta * 100 if meta else np.nan
y_, m_ = divmod(ym, 100)
ndias = calendar.monthrange(y_, m_)[1]


def meta_por_tienda_dia(meta_total, ym_, d_cur, d_prev):
    """Reparte la meta mensual por tienda y día de la semana.
    Peso = venta promedio del año anterior para esa tienda y ese día de la semana
    (si no hay, se usa el mes actual). La suma de todas las metas = meta del mes."""
    y, m = divmod(ym_, 100)
    fechas = pd.date_range(f"{y}-{m:02d}-01", periods=calendar.monthrange(y, m)[1])

    def perfil(d):
        if d.empty:
            return pd.DataFrame(columns=range(7), dtype=float)
        v = d.assign(local=d["local"].astype(str)).groupby(["local", "fecha_dia"])["venta"].sum().reset_index()
        v["dow"] = v["fecha_dia"].dt.day_of_week
        return v.groupby(["local", "dow"])["venta"].mean().unstack().reindex(columns=range(7))

    pp, pc = perfil(d_prev), perfil(d_cur)
    idx = pp.index.union(pc.index)
    prof = pp.reindex(idx).combine_first(pc.reindex(idx))
    prof = prof.apply(lambda r: r.fillna(r.mean()), axis=1)          # día sin dato -> promedio de la tienda
    prof = prof.fillna(prof.stack().mean() if prof.stack().size else 1.0).clip(lower=0)
    if prof.values.sum() <= 0:
        prof[:] = 1.0
    mat = pd.DataFrame({f: prof[f.day_of_week] for f in fechas})       # tiendas x fechas
    mat = mat / mat.values.sum() * meta_total
    tabla_dow = pd.DataFrame({DIAS[i]: mat.loc[:, [f for f in fechas if f.day_of_week == i]].mean(axis=1)
                              for i in range(7)})
    return mat.sum(axis=0), tabla_dow


meta_s, meta_tbl = meta_por_tienda_dia(meta, ym, cur, prev)


def brecha_df(dim):
    """Venta real, meta asignada y brecha de cada elemento (meta repartida según participación del año anterior)."""
    c = cur.groupby(dim, observed=True)["venta"].sum()
    p = prev.groupby(dim, observed=True)["venta"].sum()
    c.index, p.index = c.index.astype(str), p.index.astype(str)
    idx = c.index.union(p.index)
    c, p = c.reindex(idx, fill_value=0.0), p.reindex(idx, fill_value=0.0)
    if p.sum() <= 0 or not meta:
        return None
    d = pd.DataFrame({"venta": c, "meta": p / p.sum() * meta})
    d["brecha"] = d["venta"] - d["meta"]
    return d.sort_values("brecha")


def brecha(dim):
    d = brecha_df(dim)
    return None if d is None else d["brecha"]


# ----------------------------------------------------------------------------
# CAPA 1 – Resumen ejecutivo
# ----------------------------------------------------------------------------
st.title(f"Ventas Rústica – {lbl_ym(ym)}")
cabecera(1, "RESUMEN EJECUTIVO", "¿Cómo vamos?", "Cifras clave para una visión inmediata del negocio.")

b_fam = brecha("familia")
bajo = k["venta"] < meta
if bajo:
    txt = f"▼ Venta {soles(meta - k['venta'])} por debajo de la meta ({cumpl:.2f}% de cumplimiento)."
    if b_fam is not None and b_fam.iloc[0] < 0:
        txt += f" Mayor caída: <b>{esc(b_fam.index[0])}</b> ({soles_s(b_fam.iloc[0])})."
else:
    txt = f"▲ Meta alcanzada: {cumpl:.2f}% de cumplimiento (+{soles(k['venta'] - meta)})."
st.markdown(f'<div class="concl {"" if bajo else "ok"}" role="status"><b>Conclusión:</b> {txt}</div>', unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
with c1:
    tarjeta("VENTA TOTAL DEL MES", soles(k["venta"]), f"Meta: {soles(meta)}  ·  "
            + (f"faltan {soles(meta - k['venta'])}" if bajo else "meta superada"),
            barra=cumpl, estado_barra="no" if bajo else "ok")
with c2:
    dpp = cumpl - 100
    tarjeta("% CUMPLIMIENTO DE VENTA", f"{cumpl:.2f} %", "Meta definida al inicio del mes",
            f"{'▼ Bajo la meta:' if dpp < 0 else '▲ Sobre la meta:'} {abs(dpp):.2f} pp", "mal" if dpp < 0 else "bien")
with c3:
    v = var_pct(k["trans"], kp["trans"])
    tarjeta("Nº DE TRANSACCIONES", f"{k['trans']:,}", "Comandas atendidas",
            "● Sin comparativa" if v is None else f"{'▼ Baja' if v < 0 else '▲ Sube'} {abs(v):.1f}% vs. {lbl_ym(ym_prev)}",
            "alerta" if v is None else ("mal" if v < 0 else "bien"))
with c4:
    v = var_pct(k["ticket"], kp["ticket"])
    tarjeta("TICKET PROMEDIO", f"S/ {k['ticket']:,.1f}", modo_ticket,
            "● Sin comparativa" if v is None else f"{'▼ Baja' if v < 0 else '▲ Sube'} {abs(v):.1f}% vs. {lbl_ym(ym_prev)}",
            "alerta" if v is None else ("mal" if v < 0 else "bien"))

# ----------------------------------------------------------------------------
# CAPA 2 – Tendencias y comparaciones (vista GLOBAL)
# ----------------------------------------------------------------------------
st.write("")
cabecera(2, "TENDENCIAS Y COMPARACIONES", "¿Dónde está ocurriendo?",
         "Evolución, composición, comparativa y brecha para identificar áreas de foco.")
col_a, col_b, col_c = st.columns([2.6, 1.1, 0.75])


def serie_diaria(d, ym_):
    y, m = divmod(ym_, 100)
    s = d.groupby("fecha_dia")["venta"].sum()
    return s.reindex(pd.date_range(f"{y}-{m:02d}-01", periods=calendar.monthrange(y, m)[1]), fill_value=0.0)


with col_a:
    st.markdown(f"**EVOLUCIÓN DE VENTAS DEL MES – {lbl_ym(ym)}**")
    s = serie_diaria(cur, ym)
    s_vis = s[: cur["fecha_dia"].max()]
    ver_prev = st.checkbox(f"Mostrar {lbl_ym(ym_prev)}", value=False)
    metas_vis = meta_s.reindex(s_vis.index)
    mask_bajo = (s_vis.values < metas_vis.values)
    fig = go.Figure()
    fig.add_bar(x=s_vis.index[~mask_bajo], y=s_vis.values[~mask_bajo], name="Día sobre la meta ▲ (sólido)",
                marker=marcador_ok())
    fig.add_bar(x=s_vis.index[mask_bajo], y=s_vis.values[mask_bajo], name="Día bajo la meta ▼ (rayado)",
                marker=marcador_no())
    fig.add_scatter(x=meta_s.index, y=meta_s.values, mode="lines", name="Meta diaria (por tienda y día)",
                    line=dict(color=P["META"], dash="dash", shape="hv", width=2 * P["LW"]))
    fig.add_scatter(x=s_vis.index, y=s_vis.rolling(7, min_periods=1).mean(), mode="lines",
                    name="Media móvil (7 días)", line=dict(color=P["MOVIL"], width=3 * P["LW"]))
    if ver_prev and not prev.empty:
        sp = serie_diaria(prev, ym_prev).values[:ndias]
        fig.add_scatter(x=pd.date_range(s.index[0], periods=len(sp)), y=sp, mode="lines",
                        name=lbl_ym(ym_prev), line=dict(color=P["PREV"], dash="dot", width=3 * P["LW"]))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title="Venta diaria (S/)")
    fig.update_xaxes(
        tickmode="array", tickvals=list(s.index),
        ticktext=[f"{DIAS[d.day_of_week]}<br>{d.day}" for d in s.index],
        tickfont=dict(size=10, color=P["TXT"]), title=f"Día del mes – {MESES[m_ - 1]} {y_}")
    fig.update_traces(hovertemplate="%{x|%d/%m/%Y}: S/ %{y:,.0f}<extra>%{fullData.name}</extra>")
    mostrar(fig)
    ult7 = s_vis.tail(7)
    meta7 = meta_s.reindex(ult7.index)
    dif = (ult7.sum() / meta7.sum() - 1) * 100
    nb = int((ult7 < meta7).sum())
    st.caption(f"Últimos {len(ult7)} días: venta {abs(dif):.1f}% {'por debajo' if dif < 0 else 'por encima'} de la meta. "
               f"{nb} de {len(ult7)} días no alcanzaron el objetivo diario.")
    mejor, peor_d = s_vis.idxmax(), s_vis.idxmin()
    alt(f"Barras de la venta diaria de {lbl_ym(ym)}: total {soles(k['venta'])} contra una meta de {soles(meta)}. "
        f"{int(mask_bajo.sum())} de {len(s_vis)} días quedaron bajo la meta diaria (barras rayadas). "
        f"Mejor día: {mejor:%d/%m/%Y} con {soles(s_vis.max())}; día más bajo: {peor_d:%d/%m/%Y} con {soles(s_vis.min())}.")

with col_b:
    st.markdown("**COMPOSICIÓN DE LA VENTA**")
    dim_lbl = st.radio("Ver por", ["Por Familia", "Por Turno"], horizontal=True, label_visibility="collapsed")
    dim = {"Por Familia": "familia", "Por Turno": "turno"}[dim_lbl]
    if dim == "turno" and not tiene_turno:
        st.info("El dataset no tiene hora; no se puede calcular el turno.")
    else:
        g = cur.groupby(dim, observed=True)["venta"].sum().sort_values(ascending=False)
        g.index = g.index.astype(str)
        if len(g) > 7:
            g = pd.concat([g.iloc[:7], pd.Series({"Otros": g.iloc[7:].sum()})])
        pct = g / g.sum() * 100
        figc = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation="h",
                                marker=dict(color=[P["NEUTRO"] if i == "Otros" else P["OK"] for i in pct.index],
                                            line=dict(color=P["TXT"], width=P["BW"])),
                                text=[f"{v:.1f}%" for v in pct.values], textposition="outside",
                                textfont=dict(color=P["TXT"]),
                                customdata=[soles(v) for v in g.values], hovertemplate="%{y}: %{customdata}<extra></extra>"))
        figc.update_yaxes(autorange="reversed", title=dim_lbl.replace("Por ", ""))
        figc.update_xaxes(visible=False, range=[0, pct.max() * 1.5])
        mostrar(figc)
        top3 = ", ".join(f"{i} {v:.1f}%" for i, v in pct.head(3).items())
        alt(f"Barras horizontales con el porcentaje de la venta de {lbl_ym(ym)} {dim_lbl.lower()}. Las mayores: {top3}.")

with col_c:
    st.markdown("**COMPARATIVA DE VENTAS**")
    met = st.radio("Métrica", ["Venta", "Transacciones", "Ticket promedio"], horizontal=True, label_visibility="collapsed")
    clave = {"Venta": "venta", "Transacciones": "trans", "Ticket promedio": "ticket"}[met]
    etq = [lbl_ym(ym_prev), lbl_ym(ym)]
    vals = [kp[clave], k[clave]]
    cols_b = [P["NEUTRO"], P["OK"]]
    formas = ["", ""]
    if clave == "venta":
        etq.append("Meta"); vals.append(meta); cols_b.append(P["NO"]); formas.append("/")
    fmt = (lambda x: soles(x)) if clave == "venta" else ((lambda x: f"{x:,.0f}") if clave == "trans" else (lambda x: f"S/ {x:,.1f}"))
    figk = go.Figure(go.Bar(x=etq, y=[0 if pd.isna(v) else v for v in vals],
                            marker=dict(color=cols_b, line=dict(color=P["TXT"], width=P["BW"]),
                                        pattern=dict(shape=formas, fgcolor=P["TXT"], size=7, solidity=0.35, fillmode="overlay")),
                            text=[fmt(v) for v in vals], textposition="outside", textfont=dict(color=P["TXT"])))
    figk.update_yaxes(visible=False, rangemode="tozero")
    figk.update_xaxes(title=f"Período · {met}")
    mostrar(figk)
    alt(f"{met}: " + "; ".join(f"{e} {fmt(0 if pd.isna(v) else v)}" for e, v in zip(etq, vals)) + ".")
    if clave == "ticket":
        st.warning("⚠️ La data de comensales tiene problemas de calidad; evalúa el ticket con cautela.")
    v1 = var_pct(k[clave], kp[clave])
    m1 = m2 = st.container()
    m1.metric(f"Var. vs {lbl_ym(ym_prev)}", "—" if v1 is None else f"{v1:+.1f}%")
    if clave == "venta":
        vm = var_pct(k["venta"], meta)
        m2.metric("Var. vs meta", "—" if vm is None else f"{vm:+.2f}%")

# ---- Capa 2 (fila 2): descomposición de la brecha (movida desde la Capa 3) + días de la semana ----
st.write("")
col_g, col_h = st.columns([1.5, 1.1])


def cascada(d):
    """Cascada de la brecha vs meta con barras (rellenas = sobre meta, rayadas = bajo meta)."""
    b = d["brecha"]
    if len(b) > 6:
        top = b.reindex(b.abs().nlargest(6).index).sort_values()
        b = pd.concat([top, pd.Series({"Otros": b.drop(top.index).sum()})])
    cats = list(b.index) + ["Total"]
    vals_ = list(b.values)
    bases, acum = [], 0.0
    for v in vals_:
        bases.append(acum)
        acum += v
    vals_.append(acum)
    bases.append(0.0)
    fig_ = go.Figure()
    for signo, nombre, mk in ((True, "Sobre la meta ▲ (sólido)", marcador_ok()), (False, "Bajo la meta ▼ (rayado)", marcador_no())):
        ix = [i for i, v in enumerate(vals_) if (v >= 0) == signo]
        if not ix:
            continue
        fig_.add_bar(x=[cats[i] for i in ix], y=[vals_[i] for i in ix], base=[bases[i] for i in ix], name=nombre, marker=mk,
                     text=[f"{'▲' if vals_[i] >= 0 else '▼'} {soles_s(vals_[i])}" for i in ix], textposition="outside",
                     textfont=dict(color=P["TXT"]), cliponaxis=False,
                     hovertemplate="%{x}: %{text}<extra></extra>")
    fig_.update_layout(barmode="overlay")
    fig_.update_xaxes(categoryorder="array", categoryarray=cats)
    return fig_, b, acum


with col_g:
    st.markdown("**DESCOMPOSICIÓN DE LA BRECHA DE VENTA**")
    dim3_lbl = st.selectbox("Dimensión", ["Familia"] + (["Turno"] if tiene_turno else []))
    dbrecha = brecha_df(dim3_lbl.lower())
    if dbrecha is None:
        st.info(f"Se necesita venta de {lbl_ym(ym_prev)} para repartir la meta por {dim3_lbl.lower()}.")
    else:
        figw, b_plot, b_total = cascada(dbrecha)
        figw.update_yaxes(title="Diferencia vs meta (S/)")
        figw.update_xaxes(title=f"{dim3_lbl} (aporte a la brecha total)")
        mostrar(figw)
        st.caption("Supuesto: la meta se reparte según la participación de cada elemento en el año anterior. "
                   "El detalle numérico está en la Capa 3.")
        peores = [f"{i} ({soles_s(v)})" for i, v in b_plot.items() if v < 0 and i != "Otros"][:3]
        alt(f"Cascada de la brecha de venta frente a la meta por {dim3_lbl.lower()}. Resultado total: {soles_s(b_total)} "
            f"({'bajo' if b_total < 0 else 'sobre'} la meta). "
            + (f"Mayores caídas: {', '.join(peores)}." if peores else "Ningún elemento queda bajo su meta."))

with col_h:
    st.markdown("**¿QUÉ DÍAS DE LA SEMANA HAY QUE IMPULSAR?**")
    wd = s_vis.groupby(s_vis.index.day_of_week).mean().reindex(range(7))
    wm = meta_s.groupby(meta_s.index.day_of_week).mean().reindex(range(7))
    bajo_d = [(pd.notna(v) and v < t) for v, t in zip(wd, wm)]
    figd = go.Figure()
    for flag, nombre, mk in ((False, "Día sobre la meta ▲ (sólido)", marcador_ok()), (True, "Día bajo la meta ▼ (rayado)", marcador_no())):
        ix = [i for i in range(7) if bajo_d[i] == flag and pd.notna(wd.iloc[i])]
        if ix:
            figd.add_bar(x=[DIAS[i] for i in ix], y=[wd.iloc[i] for i in ix], name=nombre, marker=mk)
    figd.add_scatter(x=DIAS, y=wm.values, mode="lines+markers", name="Meta diaria",
                     line=dict(color=P["META"], dash="dash", width=2 * P["LW"]), marker=dict(symbol="diamond", size=8, color=P["META"]))
    figd.update_layout(barmode="overlay")
    figd.update_yaxes(title="Venta promedio (S/)")
    figd.update_xaxes(title="Día de la semana", categoryorder="array", categoryarray=DIAS)
    figd.update_traces(hovertemplate="%{x}: S/ %{y:,.0f}<extra>%{fullData.name}</extra>")
    mostrar(figd)
    cumpl_dow = (wd / wm).dropna()
    if not cumpl_dow.empty:
        peor = cumpl_dow.idxmin()
        st.caption(f"Día más débil vs. su meta: **{DIAS[int(peor)]}** "
                   f"(promedio {soles(wd[peor])} vs. meta {soles(wm[peor])}).")
        dias_bajo = [DIAS[i] for i in range(7) if bajo_d[i]]
        alt(f"Barras con la venta promedio por día de la semana contra su meta. "
            + (f"Días bajo la meta (rayados): {', '.join(dias_bajo)}." if dias_bajo else "Todos los días superan su meta.")
            + f" El más débil es {DIAS[int(peor)]}.")

# ----------------------------------------------------------------------------
# CAPA 3 – Diagnóstico (DETALLE)
# ----------------------------------------------------------------------------
st.write("")
cabecera(3, "DIAGNÓSTICO", "¿Por qué pasa esto?", "Detalle y tablas para investigar las causas raíz.")
col_e, col_f = st.columns(2)


def variaciones(dim, min_prev):
    c = cur.groupby(dim, observed=True)["venta"].sum()
    p = prev.groupby(dim, observed=True)["venta"].sum()
    c.index, p.index = c.index.astype(str), p.index.astype(str)
    t = pd.concat([c.rename("Venta"), p.rename("Venta ant.")], axis=1).fillna(0.0)
    t = t[t["Venta ant."] >= min_prev]
    t["Var. %"] = (t["Venta"] / t["Venta ant."] - 1) * 100
    t["Impacto S/"] = t["Venta"] - t["Venta ant."]
    return t.reset_index().rename(columns={"index": dim})


def tendencia(v):
    return "▲ Crece" if v >= 0 else "▼ Cae"


with col_e:
    st.markdown("**TOP 5 FAMILIAS – variación vs mismo período**")
    modo = st.radio("Orden", ["Mayor crecimiento", "Mayor caída"], horizontal=True, label_visibility="collapsed")
    if prev.empty:
        st.info("Sin datos del año anterior para comparar.")
    else:
        tf = variaciones("familia", prev["venta"].sum() * 0.005)
        tf = tf.sort_values("Var. %", ascending=(modo == "Mayor caída")).head(5)
        tf["Tendencia"] = tf["Var. %"].map(tendencia)
        tabla(tf[["familia", "Venta", "Var. %", "Tendencia"]], column_config={
            "familia": "Familia",
            "Venta": st.column_config.NumberColumn(format="S/ %.0f"),
            "Var. %": st.column_config.NumberColumn(format="%+.1f%%")})
        st.caption("Se excluyen familias con menos de 0.5% de la venta del año anterior.")

with col_f:
    st.markdown("**TOP 5 PRODUCTOS – mayor caída**")
    if prev.empty:
        st.info("Sin datos del año anterior para comparar.")
    else:
        min_prod = st.number_input("Venta mínima año anterior (S/)", 0, 1_000_000, 1000, step=500)
        tp = variaciones("producto", float(min_prod)).sort_values("Var. %").head(5)
        tp = tp[tp["Var. %"] < 0]
        if tp.empty:
            st.success("Ningún producto cae respecto al año anterior con ese filtro.")
        else:
            tp["Tendencia"] = tp["Var. %"].map(tendencia)
            tabla(tp[["producto", "Venta", "Var. %", "Tendencia", "Impacto S/"]], column_config={
                "producto": "Producto",
                "Venta": st.column_config.NumberColumn(format="S/ %.0f"),
                "Var. %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Impacto S/": st.column_config.ProgressColumn(
                    "Impacto", format="S/ %.0f", min_value=float(tp["Impacto S/"].min()), max_value=0.0)})
            st.caption("Candidatos a revisar su permanencia en la carta.")

st.write("")
with st.expander(f"📋 Detalle de la brecha de venta por {dim3_lbl.lower()} (tabla del gráfico de la Capa 2)", expanded=True):
    if dbrecha is None:
        st.info(f"Se necesita venta de {lbl_ym(ym_prev)} para repartir la meta por {dim3_lbl.lower()}.")
    else:
        td = dbrecha.sort_values("brecha").copy()
        td.loc["TOTAL"] = [td["venta"].sum(), td["meta"].sum(), td["brecha"].sum()]
        td["Brecha %"] = np.where(td["meta"] > 0, (td["venta"] / td["meta"].replace(0, np.nan) - 1) * 100, np.nan)
        td["Estado"] = np.where(td["brecha"] >= 0, "▲ Sobre la meta", "▼ Bajo la meta")
        td.index.name = dim3_lbl
        td = td.reset_index().rename(columns={"venta": "Venta real", "meta": "Meta asignada", "brecha": "Brecha S/"})
        tabla(td, column_config={
            "Venta real": st.column_config.NumberColumn(format="S/ %.0f"),
            "Meta asignada": st.column_config.NumberColumn(format="S/ %.0f"),
            "Brecha S/": st.column_config.NumberColumn(format="%+.0f"),
            "Brecha %": st.column_config.NumberColumn(format="%+.1f%%")})
        st.caption("Misma información del gráfico de cascada, en formato tabla (alternativa accesible).")

with st.expander("📋 Meta diaria por tienda y día de la semana", expanded=True):
    t = meta_tbl.copy()
    t.loc["TOTAL (suma de tiendas)"] = t.sum()
    t["Meta del mes"] = pd.concat([meta_tbl.mul(
        pd.Series({DIAS[i]: sum(1 for f in meta_s.index if f.day_of_week == i) for i in range(7)}), axis=1
    ).sum(axis=1), pd.Series({"TOTAL (suma de tiendas)": meta})])
    t.index.name = "Tienda"
    tabla(t.reset_index(), column_config={c: st.column_config.NumberColumn(format="S/ %.0f") for c in t.columns})
    st.caption("Meta por tienda y día de la semana, repartida según la venta promedio del año anterior "
               f"para ese día. La suma de todas las metas del mes = {soles(meta)}.")

with st.expander("Calidad de datos y detalle"):
    st.write(f"Filas del mes: {len(cur):,} · Filas {lbl_ym(ym_prev)}: {len(prev):,}")
    if mapa["comens"]:
        cm = cur["comens"].dropna()
        st.write(f"Comensales > {tope_comens} por fila: {(cm > tope_comens).sum():,} registros")

# ----------------------------------------------------------------------------
# Auditoría de accesibilidad (contraste verificable en WebAIM)
# ----------------------------------------------------------------------------
st.write("")
with st.expander("♿ Auditoría de accesibilidad – contraste WCAG (verifica cada par en WebAIM)", expanded=False):
    st.caption(f"Modo auditado: **{'Alto contraste AAA' if alto else 'Normal'}**. Texto: AAA exige 7:1. "
               "Elementos gráficos (barras, líneas, bordes): WCAG 1.4.11 exige 3:1. "
               "La razón se calcula con la misma fórmula de WebAIM; el enlace abre el verificador con los colores cargados.")
    B = BASE
    pares = [  # (elemento, color primer plano, color fondo, mínimo exigido)
        ("Texto principal sobre el fondo", B["TXT"], B["BG"], 7),
        ("Texto secundario / subtítulos", B["TXT2"], B["BG"], 7),
        ("Texto de las tarjetas KPI", B["TXT"], B["CARD"], 7),
        ("Subtítulo de las tarjetas KPI", B["TXT2"], B["CARD"], 7),
        ("Etiqueta «CAPA 1»", B["TAGTXT"], B["TAG1"], 7),
        ("Etiqueta «CAPA 2»", B["TAGTXT"], B["TAG2"], 7),
        ("Etiqueta «CAPA 3»", B["TAGTXT"], B["TAG3"], 7),
        ("Estado «bien» (▲)", B["PILL_OK"][1], B["PILL_OK"][0], 7),
        ("Estado «mal» (▼) y conclusión", B["PILL_NO"][1], B["PILL_NO"][0], 7),
        ("Estado «sin comparativa»", B["PILL_NEU"][1], B["PILL_NEU"][0], 7),
        ("Barra «cumple» (azul) vs. fondo", B["OK"], B["BG"], 3),
        ("Borde de barra «no cumple» (naranja rayado) vs. fondo", B["TXT"], B["BG"], 3),
        ("Línea de meta vs. fondo", B["META"], B["BG"], 3),
        ("Línea media móvil vs. fondo", B["MOVIL"], B["BG"], 3),
        ("Línea año anterior vs. fondo", B["PREV"], B["BG"], 3),
        ("Barra neutra («Otros», período anterior) vs. fondo", B["NEUTRO"], B["BG"], 3),
    ]
    filas = ["| Elemento | Primer plano | Fondo | Razón | Exigido | Resultado | WebAIM |", "|---|---|---|---|---|---|---|"]
    n_ok = 0
    for nombre, fg, bg, minimo in pares:
        r = contraste(fg, bg)
        ok = r >= minimo
        n_ok += ok
        nivel = ("AAA ✔" if minimo == 7 else "WCAG 1.4.11 ✔") if ok else "✖ No cumple"
        url = f"{WEBAIM}?fcolor={fg.lstrip('#')}&bcolor={bg.lstrip('#')}"
        filas.append(f"| {nombre} | `{fg}` | `{bg}` | **{r:.2f}:1** | {minimo}:1 | {nivel} | [Verificar ↗]({url}) |")
    st.markdown("\n".join(filas))
    st.markdown(f"**Resultado: {n_ok} de {len(pares)} pares cumplen.** "
                "Nota: el naranja Okabe-Ito (#E69F00) sobre blanco no llega a 3:1, por eso las barras «no cumple» "
                "llevan **rayado + borde oscuro** (el borde es lo que aporta el contraste y el rayado evita depender del color).")
