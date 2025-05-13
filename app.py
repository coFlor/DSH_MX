
import streamlit as st
import plotly.express as px
import pandas as pd
import statsmodels.api as sm

@st.cache_resource
def load_data():
    df = pd.read_csv("México_IQR.csv", index_col="tipo_habitación")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    numeric_df = df.select_dtypes(include=['int', 'float'])
    numeric_cols = numeric_df.columns

    text_df = df.select_dtypes(include=['object'])
    text_cols = text_df.columns

    cat_column = df['host_is_superhost']
    unique_categories = cat_column.unique()

    return df, numeric_cols, text_cols, unique_categories, numeric_df

df, numeric_cols, text_cols, unique_categories, numeric_df = load_data()

# Sideboard
from PIL import Image
show_sidebar = st.checkbox('Mostrar Sidebar', value=True)

view = None

if show_sidebar:
    with st.sidebar:
        st.sidebar.title("DASHBOARD")
        logo = Image.open("Flag_of_Mexico.png")
        st.sidebar.image(logo, width=100)
        st.sidebar.header("CDMX, México")

        st.sidebar.header("Sidebar")
        st.sidebar.subheader("Selección")

        view = st.sidebar.selectbox("Selecciona una vista", 
                            ["Vista 1: Categóricas & Numéricas", 
                             "Vista 2: Regresión & Correlación"])
else:
    st.write("AIRBNB VALENCIA",
            "INGENIERÍA SOFTWARE",
            "FCC BUAP ",
            "FLORES OVANDO CHRISTIAN", "EDUARDO BALLINAS BALLINAS",)
    st.write("**El Sidebar está oculto. Marca la opción para verlo.**")

#V1
if view == "Vista 1: Categóricas & Numéricas":
    
    from PIL import Image

    st.title("Análisis de Variables Categóricas")
    st.markdown("Airbnb eCDMX · Variables Categóricas")

    logo = Image.open("zocalo.jpg")
    st.image(logo, width=300)

    st.markdown("Alojamientos de Airbnb en CDMX.")

    image_dict = {
        "host_is_superhost": "imagenes/superhost.png",
        "tiene_disponibilidad": "imagenes/disponibilidad.png",
        "reservable_instantáneamente": "imagenes/reserva.png",
        "tiempo_respuesta_host": "imagenes/tiempo.png",
        "tipo_habitación": "imagenes/tipohabitacion.jpg",
        "grupo_vecindario_limpiado": "imagenes/vecindario.png",
        "fuente": "imagenes/source.png",
        "nombre": "imagenes/nombre.png"
    }

    # Mapa
    st.title("Mapa de Alojamientos en México")
    map_df = df.rename(columns={"latitud": "latitude", "longitud": "longitude"})

    map_df = map_df[

        (map_df['latitude'].between(19.2, 19.6)) & 
        (map_df['longitude'].between(-99.3, -98.9))
    ]

    st.map(map_df)

    st.title("Distribución de Categorías")

    selected_cat = st.sidebar.selectbox("Selecciona una variable categórica", text_cols)

    cat_counts = df[selected_cat].value_counts().reset_index()
    cat_counts.columns = ['Categoría', 'Frecuencia']

    cat_percents = df[selected_cat].value_counts(normalize=True).mul(100).round(2).reset_index()
    cat_percents.columns = ['Categoría', 'Porcentaje (%)']

    combined_df = pd.merge(cat_counts, cat_percents, on='Categoría')

    if len(cat_counts) > 20:
        st.warning(f"La variable {selected_cat} tiene muchas categorías.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("Imagen")
        if selected_cat in image_dict:
            try:
                st.image(image_dict[selected_cat], use_container_width=True, caption=selected_cat.replace("_", " ").capitalize())
            except:
                st.warning("No imagen")
        else:
            st.info("Sin imagen")

    with col2:
        st.markdown("Gráfico de Barras")
        fig_cat = px.bar(
            combined_df,
            x='Categoría',
            y='Frecuencia',
            color='Categoría',
            color_discrete_sequence=px.colors.qualitative.Set3,
            labels={'Frecuencia': 'Frecuencia absoluta'},
            title=f"Distribución de '{selected_cat}'"
        )
        fig_cat.update_layout(xaxis_title="Categoría", yaxis_title="Frecuencia", showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    fig_pie = px.pie(
        combined_df,
        names="Categoría",
        values="Porcentaje (%)",
        title=f"Porcentaje de '{selected_cat}'",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Tabla de Frecuencia y Porcentaje")
    st.dataframe(combined_df, use_container_width=True, height=300)

    st.title("Análisis de Variables Numéricas")
    st.markdown("Exploración de Numéricas")

    st.markdown("Distribución variables numéricas.")

    selected_num = st.sidebar.selectbox("Selecciona una variable numérica", numeric_cols)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"#### Histograma de {selected_num}")
        fig_hist = px.histogram(
            df,
            x=selected_num,
            nbins=30,
            color_discrete_sequence=["#00BFC4"],
            title=f"Distribución de {selected_num}"
        )
        fig_hist.update_layout(xaxis_title=selected_num, yaxis_title="Frecuencia", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        #st.markdown(f"#### Diagrama de Caja de `{selected_num}`")
        #fig_box = px.box(df, y=selected_num, points="outliers", color_discrete_sequence=["#F8766D"])
        #fig_box.update_layout(yaxis_title=selected_num, showlegend=False)
        #st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.markdown("#### Estadísticas Descriptivas")
        desc = df[selected_num].describe().rename({
            "count": "Cantidad",
            "mean": "Media",
            "std": "Desviación estándar",
            "min": "Mínimo",
            "25%": "Percentil 25",
            "50%": "Mediana",
            "75%": "Percentil 75",
            "max": "Máximo"
        })
        st.dataframe(desc.to_frame(name="Valor"), use_container_width=True)

    with st.expander("Vista previa del dataset"):
        st.dataframe(df[[selected_num]].dropna().head(50), use_container_width=True)


#V2
elif view == "Vista 2: Regresión & Correlación":
    st.title("Relación y Predicción entre Variables")
    from PIL import Image
    logo = Image.open("angel.jpg")
    st.image(logo, width=300)

    st.markdown("## Modelado Predictivo")

    numeric_df = numeric_df.loc[:, ~numeric_df.columns.duplicated()]

    if "reg_selected" not in st.session_state:
        st.session_state.reg_selected = None

    st.sidebar.markdown("### Selecciona el tipo de regresión:")
    if st.sidebar.button("Lineal Simple"):
        st.session_state.reg_selected = "Lineal Simple"

    if st.sidebar.button("Lineal Múltiple"):
        st.session_state.reg_selected = "Lineal Múltiple"

    if st.sidebar.button("Logística"):
        st.session_state.reg_selected = "Logística"

    reg_type = st.session_state.reg_selected

    if reg_type == "Lineal Simple":
        st.subheader("Regresión Lineal Simple")
        col1, col2 = st.columns(2)
        with col1:
            x_selected = st.selectbox("Variable independiente (X)", options=numeric_cols)
        with col2:
            y_options = [col for col in numeric_cols if col != x_selected]
            y_selected = st.selectbox("Variable dependiente (Y)", options=y_options)

        df_limpio = numeric_df[[x_selected, y_selected]].dropna()
        X = df_limpio[[x_selected]]
        y = df_limpio[y_selected]

        if not X.empty and not y.empty:
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()
            y_pred = model.predict(X_const)

            fig = px.scatter(x=X[x_selected], y=y, labels={'x': x_selected, 'y': y_selected},
                             title=f"Relación entre {x_selected} y {y_selected}")
            fig.add_scatter(x=X[x_selected], y=y_pred, mode='lines', name='Línea de regresión', line=dict(color='red'))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Resumen del Modelo")
            st.write(f"**Coeficiente de '{x_selected}':** {model.params[1]:.3f}")
            st.write(f"**R-cuadrado (R²):** {model.rsquared:.3f}")
            st.write(f"**Valor-p del coeficiente:** {model.pvalues[1]:.4f}")
            st.write(f"**Error estándar del modelo:** {model.bse[1]:.4f}")
        else:
            st.warning("No hay suficientes datos para calcular el modelo.")

    elif reg_type == "Lineal Múltiple":
        st.subheader("Regresión Lineal Múltiple")
        y_selected = st.selectbox("Variable dependiente (Y)", options=numeric_cols)
        x_selected_mult = st.multiselect("Variables independientes (X)", [col for col in numeric_cols if col != y_selected])

        if len(x_selected_mult) >= 2:
            selected_columns = [y_selected] + x_selected_mult
            df_model = numeric_df[selected_columns].dropna()

            y = df_model[y_selected]
            X = df_model[x_selected_mult]
            X_const = sm.add_constant(X)

            model = sm.OLS(y, X_const).fit()
            y_pred = model.predict(X_const)

            fig = px.scatter(x=y, y=y_pred, labels={'x': "Valores reales", 'y': "Predichos"},
                             title="Valores Reales vs. Predichos - Regresión Múltiple")
            fig.add_scatter(x=y, y=y, mode='lines', name='Línea ideal', line=dict(color='red'))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Coeficientes del Modelo")
            coef_table = pd.DataFrame({
                "Coeficiente": model.params,
                "p-valor": model.pvalues
            }).round(4)
            st.dataframe(coef_table)
        else:
            st.warning("Selecciona al menos 2 variables independientes.")

    elif reg_type == "Logística":
        st.subheader("Regresión Logística")
        st.markdown("Requiere una variable dependiente binaria.")

        bin_cols = [col for col in text_cols if df[col].nunique() == 2]
        if bin_cols:
            y_selected = st.selectbox("Variable dependiente (Y binaria)", bin_cols)
            x_selected_log = st.multiselect("Variables independientes (X)", numeric_cols)

            if x_selected_log:
                df_log = df[[y_selected] + x_selected_log].dropna()
                y = df_log[y_selected].apply(lambda x: 1 if x == df_log[y_selected].unique()[0] else 0)
                X = df_log[x_selected_log]
                X_const = sm.add_constant(X)

                model = sm.Logit(y, X_const).fit()
                st.markdown("### Resultados del modelo logístico")
                st.write(model.summary2().tables[1])

                pred_probs = model.predict(X_const)
                fig_probs = px.histogram(pred_probs, nbins=20, title="Probabilidades Predichas")
                st.plotly_chart(fig_probs, use_container_width=True)
            else:
                st.warning("Selecciona al menos una variable independiente.")
        else:
            st.warning("No hay variables categóricas binarias en el dataset.")

    st.title("Matriz de Correlación entre Variables")
    st.markdown("### Analiza las relaciones entre las variables numéricas.")
    selected_cols = st.sidebar.multiselect("Variables para la correlación", numeric_df.columns)

    if selected_cols:
        corr_matrix = numeric_df[selected_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
                        labels={'color': 'Correlación'}, title="Matriz de Correlación")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Dispersión entre dos variables")
        row, col = st.selectbox("Variables", [(x, y) for x in selected_cols for y in selected_cols if x != y])
        fig_disp = px.scatter(numeric_df, x=row, y=col, title=f"Dispersión entre {row} y {col}")
        st.plotly_chart(fig_disp, use_container_width=True)
    else:
        st.warning("Selecciona al menos una variable para ver la correlación.")
