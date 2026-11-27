# -*- coding: utf-8 -*-
"""
Dashboard de ventas Rústica – esquema de 3 capas + accesibilidad (WCAG 2.1 / Okabe-Ito)
CAPA 1: Resumen ejecutivo   (¿Cómo vamos?)
CAPA 2: Tendencias          (¿Dónde está ocurriendo?)  -> incluye la descomposición de la brecha
CAPA 3: Diagnóstico         (¿Por qué pasa esto?)       -> tops + tablas de detalle
Dataset: dataset_pos_silver.parquet (en la raíz del repositorio)

Accesibilidad aplicada (Sesión 12):
  · Paleta Okabe-Ito (sin rojo/verde): Cumple = azul, En observación = ámbar, No cumple = bermellón.
  · Nunca solo color (WCAG 1.4.1): cada estado lleva icono (● ✓ / ▲ / ■), texto y trama
    (sólido / rayado / punteado).
  · Contraste medido con la fórmula WCAG (la misma de WebAIM): texto ≥ 7:1 (AAA), gráficos ≥ 3:1.
  · Texto alternativo para cada gráfico (WCAG 1.1.1), títulos descriptivos (2.4.6).
  · Simulador de visión (Normal / Protanopia / Deuteranopia / Escala de grises).
  · Alto contraste y reducción de movimiento.
"""
import calendar
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

st.set_page_config(page_title="Dashboard de Ventas – Rústica", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------
# Constantes y paleta accesible (Okabe-Ito)
# ----------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "dataset_pos_silver.parquet"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
T1, T2 = "Turno 1 (10:00–19:00)", "Turno 2 (19:00–03:00)"

# Colores Okabe-Ito
AZUL = "#0072B2"      # Cumple
AMBAR = "#E69F00"     # En observación
BERM = "#D55E00"      # No cumple (bermellón)
CELESTE = "#56B4E9"   # barras sobre fondo oscuro
# Neutros y textos
TXT = "#1A1A1A"       # texto principal (17.4:1 sobre blanco)
TXT_SUB = "#3A3F4B"   # texto secundario (10.5:1)
GRIS_LIN = "#4A5568"  # líneas de referencia (7.5:1)
GRIS_BAR = "#6B7280"  # barras neutras (4.8:1)
FONDO = "#FFFFFF"
FONDO_SIDEBAR = "#F0F4F8"
OBS_BORDE = "#6B4700"  # borde/trama de "En observación"
MAL_BORDE = "#5C2800"  # borde de "No cumple"
AZUL_OSC = "#003A5C"
PREV_LIN = "#8A3F6E"   # año anterior (7.0:1)
# Tarjetas KPI (texto blanco sobre azul marino -> AAA)
KPI_BG = "#081F3A"
KPI_SUB = "#E6EDF5"
GRANATE = "#7A1126"    # etiquetas de capa (texto blanco -> AAA)
CONCL_BG = "#FFF3D6"
# Pills (texto oscuro sobre fondo claro del mismo matiz -> AAA)
PILL = {
    "ok": ("#D6ECF8", "#003A5C"),
    "obs": ("#FFE9B0", "#4A3200"),
    "mal": ("#FBD8C0", "#5A2300"),
    "nd": ("#E5E7EB", "#1A1A1A"),
}

# Estados: icono + texto + patrón (nunca solo color)
ESTADOS = {
    "ok": dict(icono="● ✓", txt="Cumple"),
    "obs": dict(icono="▲", txt="En observación"),
    "mal": dict(icono="■", txt="No cumple"),
    "nd": dict(icono="◆", txt="Sin dato"),
}
PATRON = {  # relleno, trama, color de la trama y borde por estado (Plotly)
    "ok": dict(shape="", bg=AZUL, fg=AZUL, line=AZUL_OSC),
    "obs": dict(shape="/", bg=AMBAR, fg=OBS_BORDE, line=OBS_BORDE),
    "mal": dict(shape=".", bg=BERM, fg="#FFFFFF", line=MAL_BORDE),
}


# ----------------------------------------------------------------------------
# Utilidades de contraste (misma fórmula WCAG 2.x que usa WebAIM)
# ----------------------------------------------------------------------------
def _lum(hex_):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contraste(fg, bg):
    a, b = sorted([_lum(fg), _lum(bg)], reverse=True)
    return (a + 0.05) / (b + 0.05)


def webaim(fg, bg):
    return f"https://webaim.org/resources/contrastchecker/?fcolor={fg.lstrip('#')}&bcolor={bg.lstrip('#')}"


# ----------------------------------------------------------------------------
# Utilidades de formato
# ----------------------------------------------------------------------------
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
    """Monto con signo explícito (el signo también comunica, no solo el color)."""
    return f"{'+' if x >= 0 else '−'}{soles(abs(x))}"


def var_pct(a, b):
    """Variación porcentual de a respecto a b (None si no se puede calcular)."""
    return None if (b is None or pd.isna(b) or b == 0) else (a / b - 1) * 100


def lbl_ym(ym):
    return f"{MESES[ym % 100 - 1]} {ym // 100}"


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
# Sidebar: filtros, meta, ticket y accesibilidad
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
UMBRAL_OBS = st.sidebar.slider("Umbral de «En observación» (% de la meta)", 50, 99, 90,
                               help="≥ 100 % = Cumple · entre este umbral y 100 % = En observación · menos = No cumple.")

st.sidebar.markdown("**Ticket promedio**")
opciones_ticket = ["Por transacción"] + (["Por comensal (depurado)"] if mapa["comens"] else [])
modo_ticket = st.sidebar.radio("Cálculo", opciones_ticket, help="El nº de comensales tiene problemas de calidad; "
                               "al depurar se excluyen comandas fuera del rango permitido.")
tope_comens = st.sidebar.number_input("Máx. comensales por comanda", 1, 1000, 30) if modo_ticket != "Por transacción" else 30

st.sidebar.markdown("**♿ Accesibilidad**")
alto_contraste = st.sidebar.checkbox("Alto contraste", help="Bordes más gruesos y fondos puros para quien necesita mayor separación entre bloques.")
reducir_mov = st.sidebar.checkbox("Reducir movimiento", help="Desactiva transiciones y animaciones.")


def estado(p):
    """Estado según % de cumplimiento (≥100 cumple · ≥umbral observación · resto no cumple)."""
    if p is None or pd.isna(p) or not np.isfinite(p):
        return "nd"
    return "ok" if p >= 100 else ("obs" if p >= UMBRAL_OBS else "mal")


# ----------------------------------------------------------------------------
# Estilos (todos los colores explícitos para poder medir el contraste)
# ----------------------------------------------------------------------------
CSS = """
<style>
.stApp {background:%FONDO%; color:%TXT%;}
[data-testid="stSidebar"] {background:%FONDO_SIDEBAR%;}
[data-testid="stCaptionContainer"], .stCaption, [data-testid="stCaptionContainer"] p {color:%TXT_SUB% !important; opacity:1 !important; font-size:.88rem !important;}
[data-testid="stWidgetLabel"] p, label p {color:%TXT% !important;}
.block-container {padding-top: 1.6rem;}
.capa {display:flex; align-items:center; flex-wrap:wrap; gap:12px; margin: 10px 0 10px 0;}
.capa .tag {background:%GRANATE%; color:#FFFFFF; font-weight:800; padding:6px 16px;
            clip-path: polygon(0 0, 88% 0, 100% 50%, 88% 100%, 0 100%); padding-right:28px;}
.capa .tit {font-weight:800; font-size:1.1rem; color:%TXT%;}
.capa .preg {color:%GRANATE%;}
.capa .sub {color:%TXT_SUB%; font-size:.9rem;}
.ptit {font-size:1rem; font-weight:800; letter-spacing:.02em; color:%TXT%; margin:.3rem 0 .4rem 0; padding:0;}
.kpi {background:%KPI_BG%; color:#FFFFFF; border:2px solid %KPI_BG%; border-radius:10px; padding:14px 16px; height:100%;}
.kpi .lbl {color:#FFFFFF; font-size:.78rem; font-weight:800; letter-spacing:.04em;}
.kpi .val {color:#FFFFFF; font-size:2rem; font-weight:800; line-height:1.15;}
.kpi .sub {color:%KPI_SUB%; font-size:.85rem;}
.pill {display:inline-block; margin-top:8px; padding:4px 10px; border-radius:6px; font-size:.85rem; font-weight:800; border:2px solid;}
.pill.ok {background:%OK_BG%; color:%OK_FG%; border-color:%OK_FG%;}
.pill.obs {background:%OBS_BG%; color:%OBS_FG%; border-color:%OBS_FG%;}
.pill.mal {background:%MAL_BG%; color:%MAL_FG%; border-color:%MAL_FG%;}
.pill.nd {background:%ND_BG%; color:%ND_FG%; border-color:%ND_FG%;}
.bar {height:12px; border-radius:6px; background:#243F5E; border:1px solid #9FB6CC; margin:8px 0 6px 0; overflow:hidden;}
.bar > div {height:100%;}
.bar > div.ok {background:%CELESTE%;}
.bar > div.obs {background:repeating-linear-gradient(45deg,%AMBAR% 0 6px,%OBS_BORDE% 6px 9px);}
.bar > div.mal {background:radial-gradient(#FFFFFF 1.4px,transparent 1.9px) 0 0/7px 7px, %BERM%;}
.mini {border:2px solid %GRIS_LIN%; border-radius:8px; padding:8px 12px; margin-top:6px; background:#FFFFFF;}
.mini .lbl {font-size:.8rem; font-weight:700; color:%TXT_SUB%;}
.mini .val {font-size:1.3rem; font-weight:800; color:%TXT%;}
.concl {border-radius:8px; padding:10px 14px; background:%CONCL_BG%; color:%TXT%; border-left:8px solid %BERM%; font-size:.95rem;}
.aviso {border-radius:6px; padding:8px 12px; background:#F3F4F6; color:%TXT%; border-left:6px solid %GRIS_LIN%; font-size:.9rem; margin:6px 0;}
.leyenda {display:flex; flex-wrap:wrap; gap:18px; align-items:center; padding:8px 12px; border:2px solid %GRIS_LIN%; border-radius:8px; color:%TXT%; font-size:.9rem; font-weight:600; margin:4px 0 8px 0;}
.leyenda b {font-weight:800;}
.sw {display:inline-block; width:16px; height:16px; vertical-align:-3px; margin-right:6px;}
.sw.ok {border-radius:50%; background:%AZUL%; border:2px solid %AZUL_OSC%;}
.sw.obs {clip-path:polygon(50% 0,100% 100%,0 100%); background:repeating-linear-gradient(45deg,%AMBAR% 0 3px,%OBS_BORDE% 3px 5px);}
.sw.mal {background:radial-gradient(#FFFFFF 1.2px,transparent 1.6px) 0 0/5px 5px, %BERM%; border:2px solid %MAL_BORDE%;}
</style>
"""
ALTO = """
<style>
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3 {color:#000000 !important;}
.kpi {background:#000000; border:4px solid #FFFFFF; outline:2px solid #000000;}
.kpi .sub {color:#FFFFFF;}
.mini, .leyenda, .aviso {border-width:3px; border-color:#000000;}
.pill {border-width:3px;}
.concl {border:3px solid #000000; border-left-width:10px;}
</style>
"""
MOV = "<style>*{animation:none !important; transition:none !important; scroll-behavior:auto !important;}</style>"

_sub = dict(FONDO=FONDO, TXT=TXT, FONDO_SIDEBAR=FONDO_SIDEBAR, TXT_SUB=TXT_SUB, GRANATE=GRANATE, KPI_BG=KPI_BG,
            KPI_SUB=KPI_SUB, CELESTE=CELESTE, AMBAR=AMBAR, OBS_BORDE=OBS_BORDE, BERM=BERM, GRIS_LIN=GRIS_LIN,
            CONCL_BG=CONCL_BG, AZUL=AZUL, AZUL_OSC=AZUL_OSC, MAL_BORDE=MAL_BORDE,
            OK_BG=PILL["ok"][0], OK_FG=PILL["ok"][1], OBS_BG=PILL["obs"][0], OBS_FG=PILL["obs"][1],
            MAL_BG=PILL["mal"][0], MAL_FG=PILL["mal"][1], ND_BG=PILL["nd"][0], ND_FG=PILL["nd"][1])
_css = CSS
for _k, _v in _sub.items():
    _css = _css.replace(f"%{_k}%", _v)
st.markdown(_css + (ALTO if alto_contraste else "") + (MOV if reducir_mov else ""), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Componentes HTML / gráficos reutilizables
# ----------------------------------------------------------------------------
def tabla(df_, **kw):
    try:
        st.dataframe(df_, hide_index=True, width="stretch", **kw)
    except TypeError:
        st.dataframe(df_, hide_index=True, use_container_width=True, **kw)


def ptit(texto):
    """Título descriptivo de cada panel (WCAG 2.4.6)."""
    st.markdown(f'<h3 class="ptit">{texto}</h3>', unsafe_allow_html=True)


def cabecera(n, titulo, pregunta, sub):
    st.markdown(
        f'<div class="capa" role="heading" aria-level="2"><span class="tag">CAPA {n}</span>'
        f'<span class="tit">{titulo} <span class="preg">{pregunta}</span></span>'
        f'<span class="sub">{sub}</span></div>',
        unsafe_allow_html=True,
    )


def pill_html(est, texto=""):
    e = ESTADOS[est]
    extra = f" · {texto}" if texto else ""
    return f'<div class="pill {est}">{e["icono"]} {e["txt"]}{extra}</div>'


def tarjeta(titulo, valor, sub="", est=None, detalle="", barra=None):
    barra_html = ""
    if barra is not None and est:
        barra_html = (f'<div class="bar"><div class="{est}" '
                      f'style="width:{min(max(barra, 0), 100):.2f}%"></div></div>')
    pill = pill_html(est, detalle) if est else ""
    st.markdown(
        f'<div class="kpi"><div class="lbl">{titulo}</div><div class="val">{valor}</div>'
        f'{barra_html}<div class="sub">{sub}</div>{pill}</div>',
        unsafe_allow_html=True,
    )


def mini(titulo, valor, est):
    st.markdown(f'<div class="mini"><div class="lbl">{titulo}</div><div class="val">{valor}</div>'
                f'{pill_html(est)}</div>', unsafe_allow_html=True)


def aviso(texto, icono="ℹ"):
    st.markdown(f'<div class="aviso"><b>{icono}</b> {texto}</div>', unsafe_allow_html=True)


def alt(texto):
    """Texto alternativo del gráfico (lectores de pantalla): tipo + datos + hallazgo. WCAG 1.1.1"""
    with st.expander("♿ Texto alternativo del gráfico"):
        st.write(texto)


def mostrar(fig):
    """Tema accesible común: fondo blanco, texto oscuro ≥ 12 px, líneas de eje con contraste."""
    fig.update_layout(
        template="plotly_white", paper_bgcolor=FONDO, plot_bgcolor=FONDO,
        font=dict(color=TXT, size=13),
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=12, color=TXT)),
    )
    fig.update_xaxes(linecolor=GRIS_LIN, tickfont=dict(color=TXT), title_font=dict(color=TXT), automargin=True)
    fig.update_yaxes(linecolor=GRIS_LIN, gridcolor="#D9DEE5", tickfont=dict(color=TXT), title_font=dict(color=TXT))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def ratios(vals, metas):
    v, m = np.asarray(vals, dtype=float), np.asarray(metas, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 0, v / m * 100, np.nan)


def etq_estado(e):
    return {"ok": "● ✓ Cumple (≥ 100 % de la meta)",
            "obs": f"▲ En observación ({UMBRAL_OBS}–99 %)",
            "mal": f"■ No cumple (< {UMBRAL_OBS} %)"}[e]


def barras_estado(fig, x, vals, metas, hover):
    """Una serie por estado, cada una con su trama: sólido / rayado / punteado (no solo color)."""
    x = list(x)
    r = ratios(vals, metas)
    vals = np.asarray(vals, dtype=float)
    est = [estado(p) for p in r]
    for e in ("ok", "obs", "mal"):
        idx = [i for i, s in enumerate(est) if s == e]
        p = PATRON[e]
        marker = dict(color=p["bg"], line=dict(color=p["line"], width=1.5))
        if p["shape"]:
            marker["pattern"] = dict(shape=p["shape"], fgcolor=p["fg"], bgcolor=p["bg"], size=6, solidity=0.45)
        fig.add_bar(
            x=[x[i] for i in idx] or [None], y=[vals[i] for i in idx] or [None], name=etq_estado(e), marker=marker,
            customdata=np.array([r[i] for i in idx]) if idx else None, hovertemplate=hover,
        )
    fig.update_layout(barmode="overlay")
    return est


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


def brecha_detalle(dim):
    """Venta real vs meta asignada (meta repartida según participación del año anterior) por elemento."""
    c = cur.groupby(dim, observed=True)["venta"].sum()
    p = prev.groupby(dim, observed=True)["venta"].sum()
    c.index, p.index = c.index.astype(str), p.index.astype(str)
    idx = c.index.union(p.index)
    c, p = c.reindex(idx, fill_value=0.0), p.reindex(idx, fill_value=0.0)
    if p.sum() <= 0 or not meta:
        return None
    asig = p / p.sum() * meta
    d = pd.DataFrame({"Venta real": c, "Meta asignada": asig})
    d["Brecha"] = d["Venta real"] - d["Meta asignada"]
    d["Cumplimiento %"] = ratios(d["Venta real"], d["Meta asignada"])
    return d.sort_values("Brecha")


def brecha(dim):
    d = brecha_detalle(dim)
    return None if d is None else d["Brecha"]


def variaciones(dim, min_prev):
    c = cur.groupby(dim, observed=True)["venta"].sum()
    p = prev.groupby(dim, observed=True)["venta"].sum()
    c.index, p.index = c.index.astype(str), p.index.astype(str)
    t = pd.concat([c.rename("Venta"), p.rename("Venta ant.")], axis=1).fillna(0.0)
    t = t[t["Venta ant."] >= min_prev]
    t["Var. %"] = (t["Venta"] / t["Venta ant."] - 1) * 100
    t["Impacto S/"] = t["Venta"] - t["Venta ant."]
    return t.reset_index().rename(columns={"index": dim})


def serie_diaria(d, ym_):
    y, m = divmod(ym_, 100)
    s = d.groupby("fecha_dia")["venta"].sum()
    return s.reindex(pd.date_range(f"{y}-{m:02d}-01", periods=calendar.monthrange(y, m)[1]), fill_value=0.0)


# ----------------------------------------------------------------------------
# Encabezado, simulador de visión y leyenda de estados
# ----------------------------------------------------------------------------
st.title(f"Ventas Rústica – {lbl_ym(ym)}")

SVG_FILTROS = (
    '<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>'
    '<filter id="sim-prot" color-interpolation-filters="linearRGB"><feColorMatrix type="matrix" '
    'values="0.152286 1.052583 -0.204868 0 0 0.114503 0.786281 0.099216 0 0 -0.003882 -0.048116 1.051998 0 0 0 0 0 1 0"/></filter>'
    '<filter id="sim-deut" color-interpolation-filters="linearRGB"><feColorMatrix type="matrix" '
    'values="0.367322 0.860646 -0.227968 0 0 0.280085 0.672501 0.047413 0 0 -0.011820 0.042940 0.968881 0 0 0 0 0 1 0"/></filter>'
    '</defs></svg>'
)
FILTRO_CSS = {"Normal": "none", "Protanopia": "url(#sim-prot)", "Deuteranopia": "url(#sim-deut)",
              "Escala de grises": "grayscale(1)"}
sim = st.radio("👁 Simulador de visión", list(FILTRO_CSS), horizontal=True,
               help="Previsualiza cómo vería este dashboard una persona con daltonismo o una impresión en blanco y negro.")
st.markdown(SVG_FILTROS + f'<style>[data-testid="stAppViewContainer"]{{filter:{FILTRO_CSS[sim]};}}</style>',
            unsafe_allow_html=True)
if sim != "Normal":
    st.caption(f"Vista simulada: **{sim}**. Los estados siguen distinguiéndose por icono, texto y trama, no solo por color.")

st.markdown(
    '<div class="leyenda" role="note"><b>Leyenda de estados:</b>'
    '<span><i class="sw ok"></i>● ✓ Cumple (≥ 100 % de la meta)</span>'
    f'<span><i class="sw obs"></i>▲ En observación ({UMBRAL_OBS}–99 %)</span>'
    f'<span><i class="sw mal"></i>■ No cumple (&lt; {UMBRAL_OBS} %)</span></div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# CAPA 1 – Resumen ejecutivo
# ----------------------------------------------------------------------------
cabecera(1, "RESUMEN EJECUTIVO", "¿Cómo vamos?", "Cifras clave para una visión inmediata del negocio.")

b_fam = brecha("familia")
if k["venta"] < meta:
    txt = f"Venta {soles(meta - k['venta'])} por debajo de la meta ({cumpl:.2f}% de cumplimiento)."
    if b_fam is not None and b_fam.iloc[0] < 0:
        txt += f" Mayor caída: <b>{b_fam.index[0]}</b> ({soles_s(b_fam.iloc[0])})."
else:
    txt = f"Meta alcanzada: {cumpl:.2f}% de cumplimiento (+{soles(k['venta'] - meta)})."
st.markdown(f'<div class="concl"><b>▲ Conclusión:</b> {txt}</div>', unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
bajo = k["venta"] < meta
e_cumpl = estado(cumpl)
with c1:
    tarjeta("VENTA TOTAL DEL MES", soles(k["venta"]),
            f"Meta: {soles(meta)}  ·  " + (f"faltan {soles(meta - k['venta'])}" if bajo else "meta superada"),
            e_cumpl, "", barra=cumpl)
with c2:
    dpp = cumpl - 100
    tarjeta("% CUMPLIMIENTO DE VENTA", f"{cumpl:.2f} %", "Meta definida al inicio del mes",
            e_cumpl, f"{'▼' if dpp < 0 else '▲'} {abs(dpp):.2f} pp vs. meta")
with c3:
    v = var_pct(k["trans"], kp["trans"])
    tarjeta("Nº DE TRANSACCIONES", f"{k['trans']:,}", "Comandas atendidas",
            "nd" if v is None else estado(100 + v),
            "sin comparativa" if v is None else f"{'▼' if v < 0 else '▲'} {v:+.1f}% vs. {lbl_ym(ym_prev)}")
with c4:
    v = var_pct(k["ticket"], kp["ticket"])
    tarjeta("TICKET PROMEDIO", f"S/ {k['ticket']:,.1f}", modo_ticket,
            "nd" if v is None else estado(100 + v),
            "sin comparativa" if v is None else f"{'▼' if v < 0 else '▲'} {v:+.1f}% vs. {lbl_ym(ym_prev)}")

# ----------------------------------------------------------------------------
# CAPA 2 – Tendencias y comparaciones (incluye la descomposición de la brecha)
# ----------------------------------------------------------------------------
st.write("")
cabecera(2, "TENDENCIAS Y COMPARACIONES", "¿Dónde está ocurriendo?",
         "Evolución, composición, comparativa y origen de la brecha para identificar áreas de foco.")
col_a, col_b, col_c = st.columns([2.6, 1.1, 0.75])

with col_a:
    ptit(f"EVOLUCIÓN DE VENTAS DEL MES – {lbl_ym(ym)}")
    s = serie_diaria(cur, ym)
    s_vis = s[: cur["fecha_dia"].max()]
    meta_vis = meta_s.reindex(s_vis.index)
    ver_prev = st.checkbox(f"Mostrar {lbl_ym(ym_prev)}", value=False)
    fig = go.Figure()
    est_dias = barras_estado(
        fig, s_vis.index, s_vis.values, meta_vis.values,
        hover="%{x|%d/%m/%Y}<br>Venta: S/ %{y:,.0f}<br>%{customdata:.0f}% de la meta del día<extra></extra>")
    media7 = s_vis.rolling(7, min_periods=1).mean()
    fig.add_scatter(x=meta_s.index, y=meta_s.values, mode="lines", name="Meta diaria (por tienda y día)",
                    line=dict(color=GRIS_LIN, dash="dash", shape="hv", width=2.5),
                    hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>")
    fig.add_scatter(x=s_vis.index, y=media7, mode="lines", name="Media móvil (7 días)",
                    line=dict(color="#000000", width=2.5), hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>")
    if ver_prev and not prev.empty:
        sp = serie_diaria(prev, ym_prev).values[:ndias]
        fig.add_scatter(x=pd.date_range(s.index[0], periods=len(sp)), y=sp, mode="lines",
                        name=lbl_ym(ym_prev), line=dict(color=PREV_LIN, dash="dot", width=3),
                        hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>")
    fig.update_yaxes(title="S/")
    fig.update_xaxes(
        tickmode="array", tickvals=list(s.index),
        ticktext=[f"{DIAS[d.day_of_week]}<br>{d.day}" for d in s.index],
        tickfont=dict(size=11, color=TXT), title=f"{MESES[m_ - 1]} {y_}")
    mostrar(fig)
    ult7 = s_vis.tail(7)
    meta7 = meta_s.reindex(ult7.index)
    dif = (ult7.sum() / meta7.sum() - 1) * 100
    nb = int((ult7 < meta7).sum())
    st.caption(f"Últimos {len(ult7)} días: venta {abs(dif):.1f}% {'por debajo' if dif < 0 else 'por encima'} de la meta. "
               f"{nb} de {len(ult7)} días no alcanzaron el objetivo diario.")
    alt(f"Barras de venta diaria de {lbl_ym(ym)}: de {len(s_vis)} días con datos, {est_dias.count('ok')} cumplieron la meta diaria, "
        f"{est_dias.count('obs')} quedaron en observación y {est_dias.count('mal')} no la cumplieron. "
        f"El día más alto fue el {s_vis.idxmax():%d/%m/%Y} con {soles(s_vis.max())} y el más bajo el {s_vis.idxmin():%d/%m/%Y} "
        f"con {soles(s_vis.min())}. Sobre las barras se dibujan la meta diaria (línea discontinua) y la media móvil de 7 días, "
        f"que cierra en {soles(media7.iloc[-1])}. En los últimos {len(ult7)} días la venta estuvo {abs(dif):.1f}% "
        f"{'por debajo' if dif < 0 else 'por encima'} de la meta.")

    with st.expander("¿Qué días de la semana hay que impulsar?", expanded=False):
        wd = s_vis.groupby(s_vis.index.day_of_week).mean().reindex(range(7))
        wm = meta_s.groupby(meta_s.index.day_of_week).mean().reindex(range(7))
        figw = go.Figure()
        est_sem = barras_estado(
            figw, DIAS, wd.values, wm.values,
            hover="%{x}<br>Venta promedio: S/ %{y:,.0f}<br>%{customdata:.0f}% de la meta<extra></extra>")
        figw.add_scatter(x=DIAS, y=wm.values, mode="lines+markers", name="Meta diaria",
                         line=dict(color=GRIS_LIN, dash="dash", width=2.5), marker=dict(symbol="diamond", size=9, color=GRIS_LIN))
        figw.update_yaxes(title="Venta promedio (S/)")
        mostrar(figw)
        cumpl_dow = (wd / wm).dropna()
        if not cumpl_dow.empty:
            peor = cumpl_dow.idxmin()
            st.caption(f"Día más débil vs. su meta: **{DIAS[int(peor)]}** "
                       f"(promedio {soles(wd[peor])} vs. meta {soles(wm[peor])}).")
            alt(f"Barras de venta promedio por día de la semana contra su meta (diamantes unidos por línea discontinua). "
                f"{est_sem.count('ok')} días cumplen, {est_sem.count('obs')} están en observación y {est_sem.count('mal')} no cumplen. "
                f"El día más débil es {DIAS[int(peor)]}: promedio {soles(wd[peor])} frente a una meta de {soles(wm[peor])}.")

with col_b:
    ptit("COMPOSICIÓN DE LA VENTA")
    dim_lbl = st.radio("Ver por", ["Por Familia", "Por Turno"], horizontal=True, label_visibility="collapsed")
    dim = {"Por Familia": "familia", "Por Turno": "turno"}[dim_lbl]
    if dim == "turno" and not tiene_turno:
        aviso("El dataset no tiene hora; no se puede calcular el turno.")
    else:
        g = cur.groupby(dim, observed=True)["venta"].sum().sort_values(ascending=False)
        g.index = g.index.astype(str)
        if len(g) > 7:
            g = pd.concat([g.iloc[:7], pd.Series({"Otros": g.iloc[7:].sum()})])
        pct = g / g.sum() * 100
        figc = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation="h",
                                marker=dict(color=[GRIS_BAR if i == "Otros" else AZUL for i in pct.index],
                                            line=dict(color=TXT, width=1)),
                                text=[f"{v:.1f}%" for v in pct.values], textposition="outside",
                                textfont=dict(size=12, color=TXT),
                                customdata=[soles(v) for v in g.values], hovertemplate="%{y}: %{customdata}<extra></extra>"))
        figc.update_yaxes(autorange="reversed")
        figc.update_xaxes(visible=False, range=[0, pct.max() * 1.25])
        mostrar(figc)
        top3 = ", ".join(f"{i} ({v:.1f}%)" for i, v in list(pct.items())[:3])
        alt(f"Barras horizontales con la participación de cada {dim_lbl.lower().replace('por ', '')} en la venta del mes, de mayor a menor. "
            f"Los tres primeros son: {top3}. Cada barra muestra su porcentaje al final.")

with col_c:
    ptit("COMPARATIVA DE VENTAS")
    met = st.radio("Métrica", ["Venta", "Transacciones", "Ticket promedio"], horizontal=True, label_visibility="collapsed")
    clave = {"Venta": "venta", "Transacciones": "trans", "Ticket promedio": "ticket"}[met]
    etq = [lbl_ym(ym_prev), lbl_ym(ym)]
    vals = [kp[clave], k[clave]]
    cols_b = [GRIS_BAR, AZUL]
    formas = ["", ""]
    if clave == "venta":
        etq.append("Meta"); vals.append(meta); cols_b.append(FONDO); formas.append("x")
    fmt = (lambda x: soles(x)) if clave == "venta" else ((lambda x: f"{x:,.0f}") if clave == "trans" else (lambda x: f"S/ {x:,.1f}"))
    figk = go.Figure(go.Bar(
        x=etq, y=[0 if pd.isna(v) else v for v in vals],
        marker=dict(color=cols_b, line=dict(color=TXT, width=1.5),
                    pattern=dict(shape=formas, fgcolor=GRIS_LIN, bgcolor=cols_b, size=6, solidity=0.35)),
        text=[fmt(v) for v in vals], textposition="outside", textfont=dict(size=12, color=TXT),
        hovertemplate="%{x}: %{text}<extra></extra>"))
    figk.update_yaxes(visible=False)
    mostrar(figk)
    alt(f"Barras que comparan {met.lower()}: " + "; ".join(f"{e}: {fmt(v)}" for e, v in zip(etq, vals)) + ".")
    if clave == "ticket":
        aviso("La data de comensales tiene problemas de calidad; evalúa el ticket con cautela.", "▲ Atención:")
    v1 = var_pct(k[clave], kp[clave])
    mini(f"Var. vs {lbl_ym(ym_prev)}", "—" if v1 is None else f"{v1:+.1f}%", "nd" if v1 is None else estado(100 + v1))
    if clave == "venta":
        vm = var_pct(k["venta"], meta)
        mini("Var. vs meta", "—" if vm is None else f"{vm:+.2f}%", "nd" if vm is None else estado(100 + vm))

# --- Descomposición de la brecha (movida desde la Capa 3: es una vista global) ---
st.write("")
col_g, col_h = st.columns([2, 1])
with col_g:
    ptit("DESCOMPOSICIÓN DE LA BRECHA DE VENTA")
    dim3_lbl = st.selectbox("Dimensión", ["Familia"] + (["Turno"] if tiene_turno else []))
    b = brecha(dim3_lbl.lower())
    if b is None:
        aviso(f"Se necesita venta de {lbl_ym(ym_prev)} para repartir la meta por {dim3_lbl.lower()}.")
    else:
        if len(b) > 6:
            top = b.reindex(b.abs().nlargest(6).index).sort_values()
            b = pd.concat([top, pd.Series({"Otros": b.drop(top.index).sum()})])
        textos = [f"{'▲' if v >= 0 else '▼'} {soles_s(v)}" for v in list(b.values) + [b.sum()]]
        figb = go.Figure(go.Waterfall(
            x=list(b.index) + ["Total"], y=list(b.values) + [0],
            measure=["relative"] * len(b) + ["total"],
            text=textos, textposition="outside", textfont=dict(size=12, color=TXT),
            connector=dict(line=dict(color=GRIS_LIN, width=1.5)),
            increasing=dict(marker=dict(color=AZUL, line=dict(color=AZUL_OSC, width=1.5))),
            decreasing=dict(marker=dict(color=BERM, line=dict(color=MAL_BORDE, width=1.5))),
            totals=dict(marker=dict(color=BERM if b.sum() < 0 else AZUL, line=dict(color=TXT, width=2)))))
        figb.update_yaxes(title="S/")
        figb.update_layout(showlegend=False)
        mostrar(figb)
        st.caption("Supuesto: la meta se reparte según la participación de cada elemento en el año anterior. "
                   "▲ aporta a la meta · ▼ resta a la meta.")
        peor_e, mejor_e = b.drop("Otros", errors="ignore").idxmin(), b.drop("Otros", errors="ignore").idxmax()
        alt(f"Gráfico de cascada de la contribución de cada {dim3_lbl.lower()} a la brecha de venta frente a la meta. "
            f"La brecha total es {soles_s(b.sum())}. La mayor caída es {peor_e} ({soles_s(b[peor_e])}) y "
            f"el mayor aporte {'positivo ' if b[mejor_e] >= 0 else ''}es {mejor_e} ({soles_s(b[mejor_e])}). "
            "Las barras que bajan restan a la meta y las que suben suman; la última columna es el total.")
with col_h:
    ptit("LECTURA DE LA BRECHA (en texto)")
    if b is not None:
        for nombre, v in b.items():
            st.markdown(f"- {'▲' if v >= 0 else '▼'} **{nombre}**: {soles_s(v)}")
        st.markdown(f"**Brecha total: {soles_s(b.sum())}**")
        st.caption("El detalle numérico por elemento (venta real, meta asignada y cumplimiento) está en la Capa 3.")

# ----------------------------------------------------------------------------
# CAPA 3 – Diagnóstico
# ----------------------------------------------------------------------------
st.write("")
cabecera(3, "DIAGNÓSTICO", "¿Por qué pasa esto?", "Detalle y análisis para investigar las causas raíz.")
col_e, col_f = st.columns(2)

with col_e:
    ptit("TOP 5 FAMILIAS – variación vs mismo período")
    modo = st.radio("Orden", ["Mayor crecimiento", "Mayor caída"], horizontal=True, label_visibility="collapsed")
    if prev.empty:
        aviso("Sin datos del año anterior para comparar.")
    else:
        tf = variaciones("familia", prev["venta"].sum() * 0.005)
        tf = tf.sort_values("Var. %", ascending=(modo == "Mayor caída")).head(5)
        tabla(tf[["familia", "Venta", "Var. %"]], column_config={
            "familia": "Familia",
            "Venta": st.column_config.NumberColumn(format="S/ %.0f"),
            "Var. %": st.column_config.NumberColumn(format="%+.1f%%")})
        st.caption("Se excluyen familias con menos de 0.5% de la venta del año anterior.")

with col_f:
    ptit("TOP 5 PRODUCTOS – mayor caída")
    if prev.empty:
        aviso("Sin datos del año anterior para comparar.")
    else:
        min_prod = st.number_input("Venta mínima año anterior (S/)", 0, 1_000_000, 1000, step=500)
        tp = variaciones("producto", float(min_prod)).sort_values("Var. %").head(5)
        tp = tp[tp["Var. %"] < 0]
        if tp.empty:
            aviso("Ningún producto cae respecto al año anterior con ese filtro.", "✓")
        else:
            tabla(tp[["producto", "Venta", "Var. %", "Impacto S/"]], column_config={
                "producto": "Producto",
                "Venta": st.column_config.NumberColumn(format="S/ %.0f"),
                "Var. %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Impacto S/": st.column_config.ProgressColumn(
                    "Impacto", format="S/ %.0f", min_value=float(tp["Impacto S/"].min()), max_value=0.0)})
            st.caption("Candidatos a revisar su permanencia en la carta.")

# --- Detalle de la brecha (tabla que respalda el gráfico de la Capa 2) ---
st.write("")
ptit(f"DETALLE DE LA BRECHA POR {dim3_lbl.upper()} (respalda el gráfico de la Capa 2)")
det = brecha_detalle(dim3_lbl.lower())
if det is None:
    aviso(f"Se necesita venta de {lbl_ym(ym_prev)} para repartir la meta por {dim3_lbl.lower()}.")
else:
    det = det.reset_index().rename(columns={"index": dim3_lbl})
    det.insert(len(det.columns), "Estado", [f"{ESTADOS[estado(p)]['icono']} {ESTADOS[estado(p)]['txt']}" for p in det["Cumplimiento %"]])
    tabla(det, column_config={
        "Venta real": st.column_config.NumberColumn(format="S/ %.0f"),
        "Meta asignada": st.column_config.NumberColumn(format="S/ %.0f"),
        "Brecha": st.column_config.NumberColumn("Brecha (S/)", format="%+.0f"),
        "Cumplimiento %": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Ordenado de la mayor caída al mayor aporte. El estado combina icono y texto (no depende del color).")

# --- Meta diaria por tienda y día de la semana (movida desde la Capa 2: es detalle) ---
st.write("")
ptit("DETALLE: META DIARIA POR TIENDA Y DÍA DE LA SEMANA")
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
# Auditoría de accesibilidad (contraste medido con la fórmula WCAG / WebAIM)
# ----------------------------------------------------------------------------
PARES = [  # (elemento, texto/gráfico, fondo, tipo)
    ("Texto principal sobre la página", TXT, FONDO, "texto"),
    ("Texto secundario y leyendas", TXT_SUB, FONDO, "texto"),
    ("Valor y etiqueta de tarjeta KPI", "#FFFFFF", KPI_BG, "texto"),
    ("Subtexto de tarjeta KPI", KPI_SUB, KPI_BG, "texto"),
    ("Etiqueta «CAPA n»", "#FFFFFF", GRANATE, "texto"),
    ("Pregunta de la capa", GRANATE, FONDO, "texto"),
    ("Conclusión", TXT, CONCL_BG, "texto"),
    ("Estado «Cumple»", PILL["ok"][1], PILL["ok"][0], "texto"),
    ("Estado «En observación»", PILL["obs"][1], PILL["obs"][0], "texto"),
    ("Estado «No cumple»", PILL["mal"][1], PILL["mal"][0], "texto"),
    ("Estado «Sin dato»", PILL["nd"][1], PILL["nd"][0], "texto"),
    ("Gráfico: barra «Cumple» vs fondo", AZUL, FONDO, "gráfico"),
    ("Gráfico: barra «No cumple» vs fondo", BERM, FONDO, "gráfico"),
    ("Gráfico: borde de «En observación» vs fondo", OBS_BORDE, FONDO, "gráfico"),
    ("Gráfico: línea de meta vs fondo", GRIS_LIN, FONDO, "gráfico"),
    ("Gráfico: año anterior vs fondo", PREV_LIN, FONDO, "gráfico"),
    ("Gráfico: barra neutra vs fondo", GRIS_BAR, FONDO, "gráfico"),
    ("Barra de progreso «Cumple» vs tarjeta", CELESTE, KPI_BG, "gráfico"),
    ("Barra de progreso «En observación» vs tarjeta", AMBAR, KPI_BG, "gráfico"),
    ("Barra de progreso «No cumple» vs tarjeta", BERM, KPI_BG, "gráfico"),
]


def nivel(r, tipo):
    if tipo == "texto":
        return "AAA ✓" if r >= 7 else ("AA ✓ (falta para AAA)" if r >= 4.5 else "✗ Falla")
    return "AA ✓ (1.4.11)" if r >= 3 else "✗ Falla"


with st.expander("♿ Auditoría de accesibilidad (WCAG 2.1) – contraste y lista de control"):
    st.markdown("Contraste calculado con la **misma fórmula WCAG 2.x** que usa "
                "[WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/). "
                "Cada enlace «Verificar» abre WebAIM con los dos colores ya cargados. "
                "Texto: AA ≥ 4.5:1, AAA ≥ 7:1 · Gráficos y bordes: AA ≥ 3:1 (criterio 1.4.11).")
    filas = ["| Elemento | Color | Fondo | Contraste | Nivel | WebAIM |", "|---|---|---|---|---|---|"]
    for nom, fg, bg, tp_ in PARES:
        r_ = contraste(fg, bg)
        filas.append(f"| {nom} | `{fg}` | `{bg}` | **{r_:.2f}:1** | {nivel(r_, tp_)} | [Verificar]({webaim(fg, bg)}) |")
    st.markdown("\n".join(filas))
    st.markdown(
        "**Lista de control (Sesión 12)**\n\n"
        "- ✓ Escala de grises: usa el simulador de visión; los estados se distinguen por trama (sólido / rayado / punteado).\n"
        "- ✓ Razón WCAG: 4.5:1 mínimo en texto (aquí ≥ 7:1) y 3:1 en gráficos.\n"
        "- ✓ Etiquetas: cada estado lleva icono + texto, nunca solo color (1.4.1).\n"
        "- ✓ Simulación de daltonismo: protanopia y deuteranopia disponibles.\n"
        "- ✓ Alt text: cada gráfico tiene su «Texto alternativo» (1.1.1).\n"
        "- ✓ Teclado: filtros, pestañas y desplegables nativos operables con Tab / Enter / Espacio (2.1.1).\n"
        "- ✓ Chunking: contenido en 3 capas y bloques; títulos descriptivos por panel (2.4.6).\n"
        "- ✓ Localización: fechas DD/MM/AAAA y montos en S/."
    )
