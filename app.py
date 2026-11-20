# -*- coding: utf-8 -*-
"""
Dashboard de ventas Rústica – esquema de 3 capas
CAPA 1: Resumen ejecutivo   (¿Cómo vamos?)
CAPA 2: Tendencias          (¿Dónde está ocurriendo?)
CAPA 3: Diagnóstico         (¿Por qué pasa esto?)
Dataset: dataset_pos_silver.parquet (en la raíz del repositorio)
"""
import calendar
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

st.set_page_config(page_title="Dashboard de Ventas – Rústica", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------
# Constantes y estilos
# ----------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "dataset_pos_silver.parquet"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
AZUL, AZUL_CL, NARANJA, ROJO, VERDE = "#1f4e9c", "#7fb2f0", "#f28c28", "#d64545", "#2e9e5b"
GRIS, GRIS_OSC = "#c3cad5", "#8a94a6"
T1, T2 = "Turno 1 (10:00–19:00)", "Turno 2 (19:00–03:00)"

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem;}
.capa {display:flex; align-items:center; gap:12px; margin: 6px 0 10px 0;}
.capa .tag {background:#c0392b; color:#fff; font-weight:700; padding:6px 16px;
            clip-path: polygon(0 0, 88% 0, 100% 50%, 88% 100%, 0 100%); padding-right:28px;}
.capa .tag.c2 {background:#e0a100;} .capa .tag.c3 {background:#2e9e5b;}
.capa .tit {font-weight:700; font-size:1.05rem;} .capa .sub {opacity:.7; font-size:.85rem;}
.kpi {border:1px solid rgba(127,127,127,.3); background:rgba(127,127,127,.07);
      border-radius:10px; padding:14px 16px; height:100%;}
.kpi .lbl {font-size:.72rem; font-weight:700; letter-spacing:.04em; opacity:.75;}
.kpi .val {font-size:2rem; font-weight:800; line-height:1.15;}
.kpi .sub {font-size:.8rem; opacity:.75;}
.pill {display:inline-block; margin-top:6px; padding:3px 10px; border-radius:6px;
       font-size:.82rem; font-weight:700;}
.pill.mal {background:rgba(214,69,69,.15); color:#d64545;}
.pill.bien {background:rgba(46,158,91,.15); color:#2e9e5b;}
.pill.alerta {background:rgba(242,140,40,.18); color:#f28c28;}
.bar {height:8px; border-radius:6px; background:rgba(127,127,127,.25); margin:8px 0 4px 0;}
.bar > div {height:8px; border-radius:6px;}
.concl {border-radius:8px; padding:8px 14px; background:rgba(242,140,40,.14); font-size:.9rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Utilidades
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


def var_pct(a, b):
    """Variación porcentual de a respecto a b (None si no se puede calcular)."""
    return None if (b is None or pd.isna(b) or b == 0) else (a / b - 1) * 100


def lbl_ym(ym):
    return f"{MESES[ym % 100 - 1]} {ym // 100}"


def mostrar(fig):
    """Compatibilidad entre versiones de Streamlit."""
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def tabla(df, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def cabecera(n, titulo, pregunta, sub):
    st.markdown(
        f'<div class="capa"><span class="tag c{n}">CAPA {n}</span>'
        f'<span class="tit">{titulo} <span style="color:#e0a100">{pregunta}</span></span>'
        f'<span class="sub">{sub}</span></div>',
        unsafe_allow_html=True,
    )


def tarjeta(titulo, valor, sub="", pill="", clase="", barra=None, color_barra=AZUL):
    barra_html = ""
    if barra is not None:
        barra_html = f'<div class="bar"><div style="width:{min(max(barra, 0), 100):.2f}%;background:{color_barra}"></div></div>'
    pill_html = f'<div class="pill {clase}">{pill}</div>' if pill else ""
    st.markdown(
        f'<div class="kpi"><div class="lbl">{titulo}</div><div class="val">{valor}</div>'
        f'{barra_html}<div class="sub">{sub}</div>{pill_html}</div>',
        unsafe_allow_html=True,
    )


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


def brecha(dim):
    """Contribución de cada elemento a la brecha vs meta (meta repartida según participación del año anterior)."""
    c = cur.groupby(dim, observed=True)["venta"].sum()
    p = prev.groupby(dim, observed=True)["venta"].sum()
    c.index, p.index = c.index.astype(str), p.index.astype(str)
    idx = c.index.union(p.index)
    c, p = c.reindex(idx, fill_value=0.0), p.reindex(idx, fill_value=0.0)
    if p.sum() <= 0 or not meta:
        return None
    return (c - p / p.sum() * meta).sort_values()


# ----------------------------------------------------------------------------
# CAPA 1 – Resumen ejecutivo
# ----------------------------------------------------------------------------
st.title(f"Ventas Rústica – {lbl_ym(ym)}")
cabecera(1, "RESUMEN EJECUTIVO", "¿Cómo vamos?", "Cifras clave para una visión inmediata del negocio.")

b_fam = brecha("familia")
if k["venta"] < meta:
    txt = f"Venta {soles(meta - k['venta'])} por debajo de la meta ({cumpl:.2f}% de cumplimiento)."
    if b_fam is not None and b_fam.iloc[0] < 0:
        txt += f" Mayor caída: <b>{b_fam.index[0]}</b> ({soles(b_fam.iloc[0])})."
else:
    txt = f"Meta alcanzada: {cumpl:.2f}% de cumplimiento (+{soles(k['venta'] - meta)})."
st.markdown(f'<div class="concl"><b>Conclusión:</b> {txt}</div>', unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
bajo = k["venta"] < meta
with c1:
    tarjeta("VENTA TOTAL DEL MES", soles(k["venta"]), f"Meta: {soles(meta)}  ·  "
            + (f"faltan {soles(meta - k['venta'])}" if bajo else "meta superada"),
            barra=cumpl, color_barra=NARANJA if bajo else VERDE)
with c2:
    dpp = cumpl - 100
    tarjeta("% CUMPLIMIENTO DE VENTA", f"{cumpl:.2f} %", "Meta definida al inicio del mes",
            f"{'▼' if dpp < 0 else '▲'} {abs(dpp):.2f} pp vs. meta", "mal" if dpp < 0 else "bien")
with c3:
    v = var_pct(k["trans"], kp["trans"])
    tarjeta("Nº DE TRANSACCIONES", f"{k['trans']:,}", "Comandas atendidas",
            "sin comparativa" if v is None else f"{'▼' if v < 0 else '▲'} {v:+.1f}% vs. {lbl_ym(ym_prev)}",
            "alerta" if v is None else ("mal" if v < 0 else "bien"))
with c4:
    v = var_pct(k["ticket"], kp["ticket"])
    tarjeta("TICKET PROMEDIO", f"S/ {k['ticket']:,.1f}", modo_ticket,
            "sin comparativa" if v is None else f"{'▼' if v < 0 else '▲'} {v:+.1f}% vs. {lbl_ym(ym_prev)}",
            "alerta" if v is None else ("mal" if v < 0 else "bien"))

# ----------------------------------------------------------------------------
# CAPA 2 – Tendencias y comparaciones
# ----------------------------------------------------------------------------
st.write("")
cabecera(2, "TENDENCIAS Y COMPARACIONES", "¿Dónde está ocurriendo?",
         "Evolución, composición y comparativa para identificar áreas de foco.")
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
    fig = go.Figure()
    fig.add_bar(x=s_vis.index, y=s_vis.values, name="Venta real",
                marker_color=[ROJO if v < mt else AZUL_CL for v, mt in zip(s_vis.values, meta_s.reindex(s_vis.index).values)])
    fig.add_scatter(x=meta_s.index, y=meta_s.values, mode="lines", name="Meta diaria (por tienda y día)",
                    line=dict(color=GRIS_OSC, dash="dash", shape="hv"))
    fig.add_scatter(x=s_vis.index, y=s_vis.rolling(7, min_periods=1).mean(), mode="lines",
                    name="Media móvil (7 días)", line=dict(color=AZUL, width=2))
    fig.add_scatter(x=[None], y=[None], mode="markers", name="Días bajo meta",
                    marker=dict(color=ROJO, symbol="square", size=10))
    if ver_prev and not prev.empty:
        sp = serie_diaria(prev, ym_prev).values[:ndias]
        fig.add_scatter(x=pd.date_range(s.index[0], periods=len(sp)), y=sp, mode="lines",
                        name=lbl_ym(ym_prev), line=dict(color=GRIS_OSC, dash="dot"))
    fig.update_yaxes(title="S/")
    fig.update_xaxes(
        tickmode="array", tickvals=list(s.index),
        ticktext=[f"{DIAS[d.day_of_week]}<br>{d.day}" for d in s.index],
        tickfont=dict(size=10), title=f"{MESES[m_ - 1]} {y_}")
    fig.update_traces(hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>")
    mostrar(fig)
    ult7 = s_vis.tail(7)
    meta7 = meta_s.reindex(ult7.index)
    dif = (ult7.sum() / meta7.sum() - 1) * 100
    nb = int((ult7 < meta7).sum())
    st.caption(f"Últimos {len(ult7)} días: venta {abs(dif):.1f}% {'por debajo' if dif < 0 else 'por encima'} de la meta. "
               f"{nb} de {len(ult7)} días no alcanzaron el objetivo diario.")

    with st.expander("¿Qué días de la semana hay que impulsar?", expanded=False):
        wd = s_vis.groupby(s_vis.index.day_of_week).mean().reindex(range(7))
        wm = meta_s.groupby(meta_s.index.day_of_week).mean().reindex(range(7))
        figw = go.Figure(go.Bar(x=DIAS, y=wd.values, name="Venta promedio",
                                marker_color=[ROJO if (pd.notna(v) and v < t) else AZUL_CL for v, t in zip(wd, wm)]))
        figw.add_scatter(x=DIAS, y=wm.values, mode="lines+markers", name="Meta diaria",
                         line=dict(color=GRIS_OSC, dash="dash"))
        figw.update_yaxes(title="Venta promedio (S/)")
        mostrar(figw)
        cumpl_dow = (wd / wm).dropna()
        if not cumpl_dow.empty:
            peor = cumpl_dow.idxmin()
            st.caption(f"Día más débil vs. su meta: **{DIAS[int(peor)]}** "
                       f"(promedio {soles(wd[peor])} vs. meta {soles(wm[peor])}).")

    with st.expander("Meta diaria por tienda y día de la semana", expanded=True):
        t = meta_tbl.copy()
        t.loc["TOTAL (suma de tiendas)"] = t.sum()
        t["Meta del mes"] = pd.concat([meta_tbl.mul(
            pd.Series({DIAS[i]: sum(1 for f in meta_s.index if f.day_of_week == i) for i in range(7)}), axis=1
        ).sum(axis=1), pd.Series({"TOTAL (suma de tiendas)": meta})])
        t.index.name = "Tienda"
        tabla(t.reset_index(), column_config={c: st.column_config.NumberColumn(format="S/ %.0f") for c in t.columns})
        st.caption("Meta por tienda y día de la semana, repartida según la venta promedio del año anterior "
                   f"para ese día. La suma de todas las metas del mes = {soles(meta)}.")

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
                                marker_color=[GRIS if i == "Otros" else AZUL for i in pct.index],
                                text=[f"{v:.1f}%" for v in pct.values], textposition="outside",
                                customdata=[soles(v) for v in g.values], hovertemplate="%{y}: %{customdata}<extra></extra>"))
        figc.update_yaxes(autorange="reversed")
        figc.update_xaxes(visible=False, range=[0, pct.max() * 1.25])
        mostrar(figc)

with col_c:
    st.markdown("**COMPARATIVA DE VENTAS**")
    met = st.radio("Métrica", ["Venta", "Transacciones", "Ticket promedio"], horizontal=True, label_visibility="collapsed")
    clave = {"Venta": "venta", "Transacciones": "trans", "Ticket promedio": "ticket"}[met]
    etq = [lbl_ym(ym_prev), lbl_ym(ym)]
    vals = [kp[clave], k[clave]]
    cols_b = [GRIS, AZUL]
    if clave == "venta":
        etq.append("Meta"); vals.append(meta); cols_b.append(GRIS_OSC)
    fmt = (lambda x: soles(x)) if clave == "venta" else ((lambda x: f"{x:,.0f}") if clave == "trans" else (lambda x: f"S/ {x:,.1f}"))
    figk = go.Figure(go.Bar(x=etq, y=[0 if pd.isna(v) else v for v in vals], marker_color=cols_b,
                            text=[fmt(v) for v in vals], textposition="outside"))
    figk.update_yaxes(visible=False)
    mostrar(figk)
    if clave == "ticket":
        st.warning("⚠️ La data de comensales tiene problemas de calidad; evalúa el ticket con cautela.")
    v1 = var_pct(k[clave], kp[clave])
    m1, m2 = st.columns(2)
    m1.metric(f"Var. vs {lbl_ym(ym_prev)}", "—" if v1 is None else f"{v1:+.1f}%")
    if clave == "venta":
        vm = var_pct(k["venta"], meta)
        m2.metric("Var. vs meta", "—" if vm is None else f"{vm:+.2f}%")

# ----------------------------------------------------------------------------
# CAPA 3 – Diagnóstico
# ----------------------------------------------------------------------------
st.write("")
cabecera(3, "DIAGNÓSTICO", "¿Por qué pasa esto?", "Detalle y análisis para investigar las causas raíz.")
col_d, col_e, col_f = st.columns([1.2, 1, 1.2])

with col_d:
    st.markdown("**DESCOMPOSICIÓN DE LA BRECHA DE VENTA**")
    dim3_lbl = st.selectbox("Dimensión", ["Familia"] + (["Turno"] if tiene_turno else []))
    b = brecha(dim3_lbl.lower())
    if b is None:
        st.info(f"Se necesita venta de {lbl_ym(ym_prev)} para repartir la meta por {dim3_lbl.lower()}.")
    else:
        if len(b) > 6:
            top = b.reindex(b.abs().nlargest(6).index).sort_values()
            b = pd.concat([top, pd.Series({"Otros": b.drop(top.index).sum()})])
        figw = go.Figure(go.Waterfall(
            x=list(b.index) + ["Total"], y=list(b.values) + [0],
            measure=["relative"] * len(b) + ["total"],
            text=[soles(v) for v in list(b.values) + [b.sum()]], textposition="outside",
            increasing=dict(marker_color=VERDE), decreasing=dict(marker_color=ROJO),
            totals=dict(marker_color=ROJO if b.sum() < 0 else VERDE)))
        figw.update_yaxes(title="S/")
        mostrar(figw)
        st.caption("Supuesto: la meta se reparte según la participación de cada elemento en el año anterior.")

with col_e:
    st.markdown("**TOP 5 FAMILIAS – variación vs mismo período**")
    modo = st.radio("Orden", ["Mayor crecimiento", "Mayor caída"], horizontal=True, label_visibility="collapsed")


    def variaciones(dim, min_prev):
        c = cur.groupby(dim, observed=True)["venta"].sum()
        p = prev.groupby(dim, observed=True)["venta"].sum()
        c.index, p.index = c.index.astype(str), p.index.astype(str)
        t = pd.concat([c.rename("Venta"), p.rename("Venta ant.")], axis=1).fillna(0.0)
        t = t[t["Venta ant."] >= min_prev]
        t["Var. %"] = (t["Venta"] / t["Venta ant."] - 1) * 100
        t["Impacto S/"] = t["Venta"] - t["Venta ant."]
        return t.reset_index().rename(columns={"index": dim})


    if prev.empty:
        st.info("Sin datos del año anterior para comparar.")
    else:
        tf = variaciones("familia", prev["venta"].sum() * 0.005)
        tf = tf.sort_values("Var. %", ascending=(modo == "Mayor caída")).head(5)
        tabla(tf[["familia", "Venta", "Var. %"]], column_config={
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
            tabla(tp[["producto", "Venta", "Var. %", "Impacto S/"]], column_config={
                "producto": "Producto",
                "Venta": st.column_config.NumberColumn(format="S/ %.0f"),
                "Var. %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Impacto S/": st.column_config.ProgressColumn(
                    "Impacto", format="S/ %.0f", min_value=float(tp["Impacto S/"].min()), max_value=0.0)})
            st.caption("Candidatos a revisar su permanencia en la carta.")

with st.expander("Calidad de datos y detalle"):
    st.write(f"Filas del mes: {len(cur):,} · Filas {lbl_ym(ym_prev)}: {len(prev):,}")
    if mapa["comens"]:
        cm = cur["comens"].dropna()
        st.write(f"Comensales > {tope_comens} por fila: {(cm > tope_comens).sum():,} registros")
