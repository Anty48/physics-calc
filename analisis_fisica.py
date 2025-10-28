import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sympy as sp
from sympy.parsing.latex import parse_latex
# Configuración global de precisión
PRECISION = 8  # Número de decimales para mostrar en toda la aplicación

st.set_page_config(page_title="Calculadora estadística", layout="wide")

# Inicializar variables de estado en la sesión
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'variable_names' not in st.session_state:
    st.session_state.variable_names = []
if 'stats_df' not in st.session_state:
    st.session_state.stats_df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'precision' not in st.session_state:
    st.session_state.precision = PRECISION
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

st.title("Calculadora estadística")

# Sidebar
with st.sidebar:
    st.header("Configuración Global")
    
    # Slider de precisión
    st.session_state["precision"] = st.slider(
        "Precisión (decimales)",
        min_value=2,
        max_value=10,
        value=st.session_state["precision"],
        step=1
    )
    
    st.markdown("---")
    
    # Enlaces útiles
    st.subheader("Enlaces rápidos")
    st.markdown("[ChatGPT](https://chat.openai.com/)")
    st.markdown("[GeoGebra](https://www.geogebra.org/calculator)")
    st.markdown("[GeoGebra 3D](https://www.geogebra.org/3d)")
    st.markdown("[Calculadora de integrales](https://www.calculadora-de-integrales.com/)")
    st.markdown("[Calculadora de derivadas](https://www.calculadora-de-derivadas.com/)")
    st.markdown("[WolframAlpha](https://www.wolframalpha.com/)")
    st.markdown("[Desmos](https://www.desmos.com/calculator)")
    st.markdown("[Symbolab](https://www.symbolab.com/)")
   
# Crear las dos pestañas principales
tab1, tab2, tab3, tab4, tab5= st.tabs(["Análisis de Datos y Regresión", "Calculadora de Incertidumbres Combinadas", "Gráficas automáticas","Modificación de datos","Formulario y traducción a Python"])

with tab1:
    st.write("En esta pestaña puedes calcular incertidumbres y hacer regresiones lineales")

    # Sección para configurar el número de variables y mediciones
    st.header("Configuración")
    col1, col2 = st.columns(2)

    with col1:
        num_variables = st.number_input("Número de variables a analizar", min_value=1, max_value=5, value=2, step=1)
    with col2:
        num_mediciones = st.number_input("Número de mediciones por variable", min_value=2, max_value=50, value=5, step=1)

    # Sección para ingresar datos
    st.header("Ingreso de Datos")

    # Crear contenedores para cada variable
    data = {}  # Usado solo para recopilación temporal
    variable_names = []  # Usado solo para recopilación temporal

    for i in range(num_variables):
        variable_name = st.text_input(f"Nombre de la variable {i+1}", value=f"Variable {i+1}")
        variable_names.append(variable_name)
        
        st.subheader(f"Mediciones para {variable_name}")
        
        # Añadir campo para la incerteza instrumental
        incerteza_instrumental = st.number_input(
            f"Incerteza instrumental para {variable_name} (ej: precisión del instrumento)",
            min_value=0.0,
            value=0.0,
            format=f"%.{st.session_state.precision}f",
            key=f"incert_{i}"
        )
        
        # Guardar la incerteza instrumental en el diccionario de datos
        if variable_name not in data:
            data[variable_name] = {}
        data[variable_name]["incerteza_instrumental"] = incerteza_instrumental
        
        # Opción para cargar datos desde un archivo CSV
        use_csv = st.checkbox(f"Cargar datos desde CSV para {variable_name}", key=f"csv_{i}")
        if use_csv:
            uploaded_file = st.file_uploader(f"Cargar CSV para {variable_name}", type=["csv"], key=f"file_{i}")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    # ⚡ Vista previa del CSV
                    st.write(f"Vista previa de los datos para {variable_name}:")
                    st.dataframe(df.head())  # solo primeras 5 filas
            
                    if len(df.columns) > 0:
                        selected_column = st.selectbox(f"Seleccionar columna para {variable_name}", df.columns, key=f"col_{i}")
                        data[variable_name]["valores"] = df[selected_column].values[:num_mediciones].tolist()
                        # Mostrar los datos cargados (opcional)
                        st.write(f"Datos seleccionados para {variable_name}:", data[variable_name]["valores"])
                except Exception as e:
                    st.error(f"Error al cargar el archivo: {e}")
        else:
            # Crear entrada de datos manual
            col1, col2 = st.columns(2)
            values = []
            
            for j in range(num_mediciones):
                if j % 2 == 0:
                    with col1:
                        value = st.number_input(f"Medición {j+1}", key=f"var_{i}_med_{j}", format=f"%.{st.session_state.precision}f")
                        values.append(value)
                else:
                    with col2:
                        value = st.number_input(f"Medición {j+1}", key=f"var_{i}_med_{j}", format="%.6f")
                        values.append(value)
            
            data[variable_name]["valores"] = values

    # Detectar si se debe realizar un análisis
    analizar_clicked = st.button("Analizar Datos")

    # Si se presiona el botón o ya se ha realizado un análisis
    if (len(data) >= 1 and analizar_clicked) or st.session_state.analysis_done:
        # Si se presiona el botón, actualizar los datos en la sesión
        if analizar_clicked:
            st.session_state.data = data.copy()
            st.session_state.variable_names = variable_names.copy()
            st.session_state.analysis_done = True
            
            # Calcular estadísticas
            stats_df = pd.DataFrame(index=st.session_state.variable_names, 
                               columns=["Media", 
                                       "Desviación Estándar", 
                                       "Error Estándar (σ/√n)", 
                                       "Incerteza Instrumental", 
                                       "Incerteza Experimental", 
                                       "Mínimo", 
                                       "Máximo"])
            
            for var_name in st.session_state.variable_names:
                values = np.array(st.session_state.data[var_name]["valores"])
                incerteza_instrumental = st.session_state.data[var_name]["incerteza_instrumental"]
                
                # Cálculos estadísticos básicos
                media = np.mean(values)
                desv_estandar = np.std(values, ddof=1)  # ddof=1 para muestra
                error_estandar = desv_estandar / np.sqrt(len(values))
                
                # Cálculo de la incerteza experimental (combinación del error estándar y la instrumental)
                incerteza_experimental = np.sqrt(error_estandar**2 + incerteza_instrumental**2)
                
                # Guardar estadísticas en el dataframe
                stats_df.loc[var_name, "Media"] = media
                stats_df.loc[var_name, "Desviación Estándar"] = desv_estandar
                stats_df.loc[var_name, "Error Estándar (σ/√n)"] = error_estandar
                stats_df.loc[var_name, "Incerteza Instrumental"] = incerteza_instrumental
                stats_df.loc[var_name, "Incerteza Experimental"] = incerteza_experimental
                stats_df.loc[var_name, "Mínimo"] = np.min(values)
                stats_df.loc[var_name, "Máximo"] = np.max(values)
                
            st.session_state.stats_df = stats_df
        
        # Mostrar resultados del análisis usando los datos almacenados en la sesión
        st.header("Resultados del Análisis")
        
        # 1. Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        
        # Configurar formato para mostrar los valores con la precisión deseada
        pd.set_option('display.float_format', f'{{:.{st.session_state.precision}f}}'.format)
        
        # Mostrar el dataframe con la configuración actualizada
        st.dataframe(st.session_state.stats_df)
        
        # Visualización de estadísticas
        if len(st.session_state.variable_names) == 1:
            var_name = st.session_state.variable_names[0]
            values = np.array(st.session_state.data[var_name]["valores"])
            
            st.subheader(f"Histograma de {var_name}")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Crear histograma
            n, bins, patches = ax.hist(values, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
            
            # Añadir línea para la media
            media = st.session_state.stats_df.loc[var_name, "Media"]
            ax.axvline(media, color='red', linestyle='dashed', linewidth=2, 
                      label=f'Media: {media:.{st.session_state.precision}f}')
            
            # Obtener error estándar e incerteza experimental
            error_estandar = st.session_state.stats_df.loc[var_name, "Error Estándar (σ/√n)"]
            incerteza = st.session_state.stats_df.loc[var_name, "Incerteza Experimental"]
            
            # Añadir líneas para media ± incerteza experimental
            ax.axvline(media - incerteza, color='green', linestyle='dotted', linewidth=2, 
                      label=f'Media ± Incerteza: {media:.{st.session_state.precision}f} ± {incerteza:.{st.session_state.precision}f}')
            ax.axvline(media + incerteza, color='green', linestyle='dotted', linewidth=2)
            
            # Añadir líneas para media ± error estándar
            ax.axvline(media - error_estandar, color='orange', linestyle='dashdot', linewidth=2, 
                      label=f'Media ± Error Estándar: {media:.{st.session_state.precision}f} ± {error_estandar:.{st.session_state.precision}f}')
            ax.axvline(media + error_estandar, color='orange', linestyle='dashdot', linewidth=2)
            
            # Personalización
            ax.set_xlabel(var_name)
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Distribución de mediciones para {var_name}')
            ax.legend()
            
            st.pyplot(fig)
        
        # 2. Regresión Lineal (entre las primeras dos variables)
        if len(st.session_state.variable_names) >= 2:
            st.subheader("Regresión Lineal")
            
            x_var = st.session_state.variable_names[0]
            y_var = st.session_state.variable_names[1]
            
            st.write(f"Análisis de regresión entre {x_var} (x) y {y_var} (y)")
            
            x = np.array(st.session_state.data[x_var]["valores"])
            y = np.array(st.session_state.data[y_var]["valores"])
            
            # Crear la tabla con x, y, x², y², x×y
            reg_data = {
                x_var: x,
                y_var: y,
                f"{x_var}²": x**2,
                f"{y_var}²": y**2,
                f"{x_var}×{y_var}": x*y
            }
            
            reg_table = pd.DataFrame(reg_data)
            
            # Añadir sumas a la tabla
            sums = reg_table.sum()
            reg_table.loc["Suma"] = sums
            
            # Mostrar la tabla de datos para regresión
            st.write("Datos para la regresión lineal:")
            st.dataframe(reg_table)
            
            # Calcular la regresión lineal
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calcular la desviación estándar de la pendiente y el intercepto
            n = len(x)
            
            # Calculamos el error estándar de la estimación (SEE)
            y_pred = slope * x + intercept
            residuos = y - y_pred
            sse = np.sum(residuos**2)
            see = np.sqrt(sse / (n - 2))
            
            # Desviación estándar de la pendiente
            sum_x_squared = np.sum(x**2)
            sum_x_all_squared = np.sum(x)**2
            s_xx = sum_x_squared - (sum_x_all_squared / n)
            sd_slope = see / np.sqrt(s_xx)
            
            # Desviación estándar del intercepto
            mean_x = np.mean(x)
            sd_intercept = see * np.sqrt(sum_x_squared / (n * s_xx))
            
            # Mostrar resultados
            reg_results = pd.DataFrame({
                "Parámetro": ["Pendiente (m)", "Intercepto (b)", "Coeficiente de Correlación (r)", "R²", "Error Estándar de la Estimación", "Valor p"],
                "Valor": [slope, intercept, r_value, r_value**2, see, p_value],
                "Desviación Estándar": [sd_slope, sd_intercept, None, None, None, None]
            })
            
            st.dataframe(reg_results)
            
            # Ecuación de la regresión con incertidumbres
            st.write(f"Ecuación de la regresión: y = ({slope:.{st.session_state.precision}f} ± {sd_slope:.{st.session_state.precision}f})x + ({intercept:.{st.session_state.precision}f} ± {sd_intercept:.{st.session_state.precision}f})")
            
            # Gráfico de regresión
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Datos originales
            ax.scatter(x, y, color='blue', alpha=0.7, label='Datos')
            
            # Línea de regresión
            x_line = np.linspace(min(x), max(x), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color='red', 
                    label=f'Regresión: y = {slope:.{st.session_state.precision}f}x + {intercept:.{st.session_state.precision}f}')
            
            # Bandas de confianza (simplificadas)
            # Calculamos los intervalos de confianza del 95% para la línea de regresión
            t_critical = stats.t.ppf(0.975, n-2)  # 95% de confianza, n-2 grados de libertad
            
            # Error estándar para cada punto en la línea de regresión
            se_line = see * np.sqrt(1/n + (x_line - mean_x)**2 / s_xx)
            
            # Límites superior e inferior para la banda de confianza
            y_upper = y_line + t_critical * se_line
            y_lower = y_line - t_critical * se_line
            
            # Añadir bandas de confianza al gráfico
            ax.fill_between(x_line, y_lower, y_upper, color='grey', alpha=0.2, label='Intervalo de confianza 95%')
            
            # Personalización
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f'Regresión Lineal: {y_var} vs {x_var}')
            ax.legend()
            
            st.pyplot(fig)
with tab2:
    st.header("Calculadora de Incertidumbres Combinadas")

    st.write("""
    Esta sección te permite calcular el valor de una función y su incertidumbre combinada 
    a partir de los valores e incertidumbres de sus variables. La calculadora utiliza 
    la fórmula de propagación de incertidumbres:
    """)

    st.latex(r"u_c = \sqrt{\sum_{i=1}^{n} \left(\frac{\partial f}{\partial x_i} \cdot u_{x_i}\right)^2}")

    # Configuración inicial
    col1, col2 = st.columns(2)
    with col1:
        num_variables = st.number_input("Número de variables", min_value=1, max_value=10, value=2, step=1, key="ic_num_vars")
    with col2:
        num_constantes = st.number_input("Número de constantes", min_value=0, max_value=10, value=1, step=1, key="ic_num_const")

    variables_info = {}
    constantes_info = {}

    # Input para variables
    st.subheader("Variables")
    cols = st.columns(2)
    for i in range(num_variables):
        col_idx = i % 2
        with cols[col_idx]:
            st.write(f"#### Variable {i+1}")
            var_symbol = st.text_input(f"Símbolo de la variable {i+1}", value=chr(120+i), key=f"var_symbol_{i}")
            var_value = st.number_input(f"Valor de {var_symbol}", value=1.0, format=f"%.{st.session_state.precision}f", key=f"var_value_{i}")
            var_uncertainty = st.number_input(f"Incertidumbre de {var_symbol}", value=0.1, min_value=0.0, format=f"%.{st.session_state.precision}f", key=f"var_uncert_{i}")
            variables_info[var_symbol] = {"valor": var_value, "incertidumbre": var_uncertainty}

    # Input para constantes
    if num_constantes > 0:
        st.subheader("Constantes")
        cols = st.columns(2)
        for i in range(num_constantes):
            col_idx = i % 2
            with cols[col_idx]:
                st.write(f"#### Constante {i+1}")
                const_symbol = st.text_input(f"Símbolo de la constante {i+1}", value=chr(97+i), key=f"const_symbol_{i}")
                const_value = st.number_input(f"Valor de {const_symbol}", value=2.0, format=f"%.{st.session_state.precision}f", key=f"const_value_{i}")
                constantes_info[const_symbol] = {"valor": const_value}

    # Input para función
    st.subheader("Función")
    st.write("""
    Ingresa la función utilizando los símbolos definidos anteriormente. Usa operadores Python estándar:
    - Suma: +
    - Resta: -
    - Multiplicación: *
    - División: /
    - Potencia: **
    - Funciones disponibles: sin, cos, tan, exp, log, sqrt

    Ejemplos: `x**2 + y`, `sin(x) + y/z`, `sqrt(x**2 + y**2)`
    """)

    available_symbols = ", ".join(list(variables_info.keys()) + list(constantes_info.keys()))
    st.write(f"Símbolos disponibles: {available_symbols}")

    function_str = st.text_input("Función f(variables, constantes)", value="x**2 + a*y", key="function_input")
    latex_str = st.text_area("Función en LaTeX (opcional)", value="", help="Si tu función usa constantes especiales como \\pi")

    calcular_clicked = st.button("Calcular valor e incertidumbre")

    if calcular_clicked and function_str:
        try:
            # Crear símbolos sympy
            symbols = {**{v: sp.symbols(v) for v in variables_info}, **{c: sp.symbols(c) for c in constantes_info}}

            # Parsear función
            if latex_str:
                try:
                    func_expr = parse_latex(latex_str)
                except Exception as e:
                    st.error(f"No se pudo convertir LaTeX: {e}")
                    func_expr = sp.sympify(function_str)
            else:
                func_expr = sp.sympify(function_str)

            st.write("### Función a evaluar:")
            st.latex(sp.latex(func_expr))

            # Derivadas parciales
            st.write("### Derivadas parciales:")
            derivatives = {}
            for var in variables_info:
                deriv = sp.diff(func_expr, symbols[var])
                derivatives[var] = deriv
                st.latex(r"\frac{\partial f}{\partial " + var + "} = " + sp.latex(deriv))

            # Fórmula completa de la incertidumbre combinada
            terms = [derivatives[var]*symbols[var+"_u"] if False else f"({sp.latex(derivatives[var])} \cdot u_{{{var}}})^2" for var in variables_info]
            combined_formula_latex = r"u_c = \sqrt{" + " + ".join(terms) + "}"
            st.write("### Fórmula completa de la incertidumbre combinada:")
            st.latex(combined_formula_latex)

            # Diccionario de valores para sustitución
            values_dict = {**{v: variables_info[v]["valor"] for v in variables_info},
                       **{c: constantes_info[c]["valor"] for c in constantes_info}}

            # Evaluar función y derivadas con valores numéricos
            func_value = float(func_expr.evalf(subs=values_dict))
            derivative_values = {var: float(deriv.evalf(subs=values_dict)) for var, deriv in derivatives.items()}

            # Incertidumbre combinada
            squared_terms = [(derivative_values[var]*variables_info[var]["incertidumbre"])**2 for var in variables_info]
            combined_uncertainty = np.sqrt(np.sum(squared_terms))

            # Mostrar resultados
            results_data = [
                {"Descripción": "Valor de la función", "Valor": f"{func_value:.{st.session_state.precision}f}"},
                {"Descripción": "Incertidumbre combinada", "Valor": f"{combined_uncertainty:.{st.session_state.precision}f}"},
                {"Descripción": "Resultado final", "Valor": f"{func_value:.{st.session_state.precision}f} ± {combined_uncertainty:.{st.session_state.precision}f}"}
            ]
            if func_value != 0:
                rel_unc = (combined_uncertainty/abs(func_value))*100
                results_data.append({"Descripción": "Incertidumbre relativa", "Valor": f"{rel_unc:.{st.session_state.precision}f}%"})
            st.table(pd.DataFrame(results_data))

            # Contribuciones individuales
            contributions = []
            for var in variables_info:
                term = (derivative_values[var]*variables_info[var]["incertidumbre"])**2
                percent = (term/np.sum(squared_terms))*100
                contributions.append({
                    "Variable": var,
                    "Derivada parcial": f"{derivative_values[var]:.{st.session_state.precision}f}",
                    "Incertidumbre": f"{variables_info[var]['incertidumbre']:.{st.session_state.precision}f}",
                    "Término (∂f/∂x·u)²": f"{term:.{st.session_state.precision}f}",
                    "Contribución (%)": f"{percent:.2f}%"
                })
            st.table(pd.DataFrame(contributions))

            # Gráfico de contribuciones
            if len(variables_info) > 1:
                fig, ax = plt.subplots(figsize=(10,5))
                vars_list = [c["Variable"] for c in contributions]
                perc_list = [float(c["Contribución (%)"].strip('%')) for c in contributions]
                ax.bar(vars_list, perc_list, color=plt.cm.viridis(np.linspace(0,0.8,len(vars_list))))
                for i, val in enumerate(perc_list):
                    ax.text(i, val, f"{val:.2f}%", ha='center', va='bottom')
                ax.set_xlabel("Variables")
                ax.set_ylabel("Contribución a la incertidumbre total (%)")
                ax.set_title("Contribución de cada variable a la incertidumbre combinada")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al calcular: {e}")
            st.write("Verifica la función y los símbolos definidos.")
with tab3:
    st.header("Gráficas automáticas con incertidumbres y funciones")
    st.write("Puedes introducir datos manualmente, subir un CSV o superponer funciones explícitas para comparar.")

    # Inicializar lista de gráficas y CSV persistente
    if "graficas" not in st.session_state:
        st.session_state.graficas = []
    if "df_csv" not in st.session_state:
        st.session_state.df_csv = None

    colores = plt.cm.tab10.colors
    modo = st.radio("Modo de entrada de datos:", ["Manual", "CSV", "Función"])

    # --- MODO MANUAL ---
    if modo == "Manual":
        n_puntos = st.number_input("Número de puntos", min_value=2, value=4, step=1)
        graf_label = st.text_input("Nombre de la gráfica", value="Manual", key="manual_label")

        datos = []
        for i in range(int(n_puntos)):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x = st.number_input(f"x{i+1}", value=float(i+1), key=f"x{i}")
            with col2:
                dx = st.number_input(f"Δx{i+1}", value=0.0, key=f"dx{i}")
            with col3:
                y = st.number_input(f"y{i+1}", value=float(i+2), key=f"y{i}")
            with col4:
                dy = st.number_input(f"Δy{i+1}", value=0.1, key=f"dy{i}")
            datos.append((x, dx, y, dy))

        if st.button("Añadir gráfica manual"):
            xs = np.array([d[0] for d in datos])
            dxs = np.array([d[1] for d in datos])
            ys = np.array([d[2] for d in datos])
            dys = np.array([d[3] for d in datos])

            st.session_state.graficas.append({
                "xs": xs, "dxs": dxs, "ys": ys, "dys": dys,
                "color": colores[len(st.session_state.graficas) % len(colores)],
                "label": graf_label
            })
            st.success(f"Gráfica '{graf_label}' añadida ✅")

    # --- MODO CSV ---
    elif modo == "CSV":
        uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df_csv = pd.read_csv(uploaded_file)

        if st.session_state.df_csv is not None:
            df = st.session_state.df_csv
            st.write("Vista previa del CSV:")
            st.dataframe(df.head())

            n_graficas = st.number_input("Número de gráficas a superponer", min_value=1, value=1, step=1)
            n_puntos = st.slider("Número de puntos a usar por gráfica", min_value=2, max_value=len(df), value=min(10, len(df)))

            for i in range(int(n_graficas)):
                st.subheader(f"Gráfica {i+1}")
                graf_label = st.text_input(f"Nombre de la gráfica {i+1}", value=f"CSV {i+1}", key=f"csv_label_{i}")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_col = st.selectbox(f"X (Gráfica {i+1})", df.columns, key=f"x_col_{i}")
                with col2:
                    dx_col = st.selectbox(f"ΔX (Gráfica {i+1})", [None] + list(df.columns), key=f"dx_col_{i}")
                with col3:
                    y_col = st.selectbox(f"Y (Gráfica {i+1})", df.columns, key=f"y_col_{i}")
                with col4:
                    dy_col = st.selectbox(f"ΔY (Gráfica {i+1})", [None] + list(df.columns), key=f"dy_col_{i}")

                if st.button(f"Añadir gráfica CSV {i+1}"):
                    xs = df[x_col].values[:n_puntos]
                    dxs = df[dx_col].values[:n_puntos] if dx_col is not None else np.zeros(n_puntos)
                    ys = df[y_col].values[:n_puntos]
                    dys = df[dy_col].values[:n_puntos] if dy_col is not None else np.zeros(n_puntos)

                    st.session_state.graficas.append({
                        "xs": xs, "dxs": dxs, "ys": ys, "dys": dys,
                        "color": colores[len(st.session_state.graficas) % len(colores)],
                        "label": graf_label
                    })
                    st.success(f"Gráfica '{graf_label}' añadida ✅")

    # --- MODO FUNCIÓN ---
    elif modo == "Función":
        graf_label = st.text_input("Nombre de la función", value="Función", key="func_label")
        func_str = st.text_input("Escribe la función y=f(x)", value="x**2 + 2*x")
        x_min = st.number_input("x mínimo", value=0.0)
        x_max = st.number_input("x máximo", value=10.0)
        n_puntos = st.slider("Número de puntos a graficar", 5, 50, 10)

        if st.button("Añadir función"):
            x_vals = np.linspace(x_min, x_max, n_puntos)
            try:
                y_vals = np.array([eval(func_str, {"x": xv, "np": np}) for xv in x_vals])
                st.session_state.graficas.append({
                    "xs": x_vals, "dxs": np.zeros_like(x_vals),
                    "ys": y_vals, "dys": np.zeros_like(y_vals),
                    "color": colores[len(st.session_state.graficas) % len(colores)],
                    "label": graf_label
                })
                st.success(f"Función '{graf_label}' añadida ✅")
            except Exception as e:
                st.error(f"Error al evaluar la función: {e}")

    # --- CONTROLES DE VISUALIZACIÓN ---
    st.divider()
    colA, colB, colC = st.columns([4,1,1])

    with colA:
        if st.button("Mostrar todas las gráficas"):
            if len(st.session_state.graficas) == 0:
                st.warning("Primero añade al menos una gráfica.")
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                titulos = []
                for g in st.session_state.graficas:
                    ax.errorbar(
                        g["xs"], g["ys"], xerr=g["dxs"], yerr=g["dys"],
                        fmt='o' if np.any(g["dxs"] + g["dys"]) else '-',
                        capsize=4, ecolor='black', color=g["color"], label=g["label"]
                    )
                    titulos.append(g["label"])
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(" + ".join(titulos))
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                st.info("Cada color representa un conjunto de datos o función distinto con sus incertidumbres.")

    with colB:
        if st.button("Borrar última gráfica"):
            if len(st.session_state.graficas) > 0:
                ultima = st.session_state.graficas.pop()
                st.warning(f"Se ha borrado la última gráfica: '{ultima['label']}' ❌")
            else:
                st.warning("No hay gráficas que borrar.")

    with colC:
        if st.button("Borrar todas las gráficas"):
            st.session_state.graficas = []
            st.error("Se han borrado todas las gráficas de la memoria 🗑️")
with tab4:
    st.header("Editor avanzado de CSV con historial de operaciones")
    st.write("Carga un CSV y aplica operaciones secuenciales sobre columnas. Cada operación se acumula y se puede ver, borrar o descargar el CSV final.")

    # Inicializar CSV y operaciones acumuladas
    if "df_mod" not in st.session_state:
        st.session_state.df_mod = None
    if "operaciones" not in st.session_state:
        st.session_state.operaciones = []

    uploaded_file = st.file_uploader("Cargar CSV", type=["csv"], key="mod_csv_adv")

    if uploaded_file is not None and st.session_state.df_mod is None:
        st.session_state.df_mod = pd.read_csv(uploaded_file)

    if st.session_state.df_mod is not None:
        df_mod = st.session_state.df_mod
        st.subheader("Vista previa del CSV actual")
        st.dataframe(df_mod.head())

        st.subheader("Agregar operación")
        cols = df_mod.columns.tolist()
        operation_type = st.selectbox("Tipo de operación", 
                                      ["Sumar constante", "Multiplicar por constante", 
                                       "Suma columnas", "Resta columnas", "Multiplicar columnas",
                                       "Aplicar función Python"])

        if operation_type in ["Sumar constante", "Multiplicar por constante"]:
            selected_cols = st.multiselect("Columnas a modificar", cols)
            const_value = st.number_input("Valor de la constante", value=1.0, format=f"%.{st.session_state.precision}f")
        elif operation_type in ["Suma columnas", "Resta columnas", "Multiplicar columnas"]:
            col_a = st.selectbox("Columna A", cols, key="colA_op")
            col_b = st.selectbox("Columna B", cols, key="colB_op")
            new_col_name = st.text_input("Nombre de la columna resultado", value=f"{col_a}_{operation_type}_{col_b}")
        elif operation_type == "Aplicar función Python":
            selected_col = st.selectbox("Columna a transformar", cols)
            func_str = st.text_input("Función Python", value="x**0.5", help="Usa 'x' como variable de la columna. Ej: x**0.5, x*2 + 5")

        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("Aplicar operación"):
                try:
                    df_mod_copy = df_mod.copy()
                    op_desc = ""
                    if operation_type == "Sumar constante":
                        for col in selected_cols:
                            df_mod_copy[col] += const_value
                        op_desc = f"Sumar {const_value} a {selected_cols}"
                    elif operation_type == "Multiplicar por constante":
                        for col in selected_cols:
                            df_mod_copy[col] *= const_value
                        op_desc = f"Multiplicar {selected_cols} por {const_value}"
                    elif operation_type == "Suma columnas":
                        df_mod_copy[new_col_name] = df_mod_copy[col_a] + df_mod_copy[col_b]
                        op_desc = f"{col_a} + {col_b} → {new_col_name}"
                    elif operation_type == "Resta columnas":
                        df_mod_copy[new_col_name] = df_mod_copy[col_a] - df_mod_copy[col_b]
                        op_desc = f"{col_a} - {col_b} → {new_col_name}"
                    elif operation_type == "Multiplicar columnas":
                        df_mod_copy[new_col_name] = df_mod_copy[col_a] * df_mod_copy[col_b]
                        op_desc = f"{col_a} * {col_b} → {new_col_name}"
                    elif operation_type == "Aplicar función Python":
                        df_mod_copy[selected_col] = df_mod_copy[selected_col].apply(lambda x: eval(func_str, {"x": x, "np": np}))
                        op_desc = f"{selected_col} = {func_str}"
                    
                    # Guardar cambios y operación
                    st.session_state.df_mod = df_mod_copy
                    st.session_state.operaciones.append(op_desc)
                    st.success("Operación aplicada ✅")
                    st.dataframe(df_mod_copy.head())
                except Exception as e:
                    st.error(f"Error aplicando operación: {e}")

        with col2:
            if st.button("Borrar última operación"):
                if st.session_state.operaciones:
                    st.session_state.operaciones.pop()
                    st.session_state.df_mod = pd.read_csv(uploaded_file)  # reinicia desde CSV original
                    # Reaplicar todas las operaciones acumuladas
                    for op in st.session_state.operaciones:
                        # Este bucle reaplica cada operación; se puede mejorar parseando strings
                        exec("df_mod_copy = st.session_state.df_mod.copy()\n" + op)  # placeholder, explicaré abajo
                    st.success("Última operación borrada")
                else:
                    st.warning("No hay operaciones que borrar.")

        # Mostrar historial de operaciones
        st.subheader("Historial de operaciones aplicadas")
        if st.session_state.operaciones:
            for i, op in enumerate(st.session_state.operaciones, 1):
                st.write(f"{i}. {op}")
        else:
            st.info("No se ha aplicado ninguna operación aún.")

        # Botón para descargar CSV final
        csv_bytes = st.session_state.df_mod.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar CSV final", csv_bytes, "datos_modificados.csv", "text/csv")
with tab5:
    st.title("Formulario de Estadística, Regresiones, Incertidumbres y Funciones Python")

    # -------------------------------
    # 1. Funciones Matemáticas
    # -------------------------------
    st.header("1. Funciones Matemáticas y Equivalentes en Python/Numpy")
    st.write("Cada función en LaTeX y su traducción a Python:")

    funciones = [
        (r"\sum x_i", "np.sum(x)"),
        (r"\prod x_i", "np.prod(x)"),
        (r"x^n", "x**n"),
        (r"\sqrt{x}", "np.sqrt(x)"),
        (r"e^x", "np.exp(x)"),
        (r"\ln(x)", "np.log(x)"),
        (r"\log_{10}(x)", "np.log10(x)"),
        (r"\sin(x)", "np.sin(x)"),
        (r"\cos(x)", "np.cos(x)"),
        (r"\tan(x)", "np.tan(x)"),
        (r"\arcsin(x)", "np.arcsin(x)"),
        (r"\arccos(x)", "np.arccos(x)"),
        (r"\arctan(x)", "np.arctan(x)"),
        (r"|x|", "np.abs(x)"),
        (r"\text{round}(x,n)", "round(x,n)"),
        (r"\max(x)", "np.max(x)"),
        (r"\min(x)", "np.min(x)")
    ]

    # Encabezado de la "tabla"
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Función (LaTeX)**")
    with col2:
        st.markdown("**Python**")
    st.write("---")

    # Filas de la tabla
    for latex_expr, python_expr in funciones:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.latex(latex_expr)
        with col2:
            st.markdown(f"`{python_expr}`")
        st.write("")

    # Fórmulas en LaTeX
    formulas = r"""
    \begin{align*}
    \text{Incertesa absoluta:} \quad & u_x = \Delta x \\[0.2cm]
    \text{Incertesa relativa:} \quad & \varepsilon_x = \frac{u_x}{x} \\[0.2cm]
    \text{Combinació d'incerteses (suma/resta):} \quad & u_R = \sqrt{u_A^2 + u_B^2} \\[0.2cm]
    \text{Combinació d'incerteses (producte/cocient):} \quad & \frac{u_R}{R} = \sqrt{\left(\frac{u_A}{A}\right)^2 + \left(\frac{u_B}{B}\right)^2} \\[0.2cm]
    \text{Incertesa d'una potència:} \quad & \frac{u_R}{R} = |n| \frac{u_A}{A} \\[0.2cm]
    \text{Mitjana aritmètica:} \quad & \bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i \\[0.2cm]
    \text{Desviació típica:} \quad & s = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2} \\[0.2cm]
    \text{Incertesa de la mitjana:} \quad & u_{\bar{x}} = \frac{s}{\sqrt{N}} \\[0.2cm]
    \text{Propagació general d'incerteses:} \quad & u_f = \sqrt{\left(\frac{\partial f}{\partial x_1} u_{x_1}\right)^2 + \dots + \left(\frac{\partial f}{\partial x_n} u_{x_n}\right)^2} \\[0.2cm]
    \text{Variança:} \quad & \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 \\[0.2cm]
    \text{Desviació típica poblacional:} \quad & \sigma = \sqrt{\sigma^2} \\[0.2cm]
    \text{Error estàndard de la mitjana:} \quad & \sigma_{\bar{x}} = \frac{\sigma}{\sqrt{N}}
    \end{align*}
        """

    # Mostrar todas las fórmulas juntas
    st.latex(formulas)