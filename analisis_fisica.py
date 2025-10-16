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

# Añadir control para la precisión
with st.sidebar:
    st.header("Configuración Global")
    precision = st.slider(
        "Precisión (número de decimales)",
        min_value=2,
        max_value=10,
        value=st.session_state.precision,
        step=1
    )
    if precision != st.session_state.precision:
        st.session_state.precision = precision
        st.experimental_rerun()

# Crear las dos pestañas principales
tab1, tab2, tab3 = st.tabs(["Análisis de Datos y Regresión", "Calculadora de Incertidumbres Combinadas", "Gráficas automáticas"])

# Contenido de la primera pestaña: Análisis de Datos y Regresión
with tab1:
    st.write("En esta pestaña puedes calcular incertidumbres y hacer regresiones lineales")

    # Sección para configurar el número de variables y mediciones
    st.header("Configuración")
    col1, col2 = st.columns(2)

    with col1:
        num_variables = st.number_input("Número de variables a analizar", min_value=1, max_value=5, value=2, step=1)
    with col2:
        num_mediciones = st.number_input("Número de mediciones por variable", min_value=5, max_value=50, value=5, step=1)

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
            ax.grid(True, linestyle='--', alpha=0.7)
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
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            st.pyplot(fig)

# Contenido de la segunda pestaña: Calculadora de Incertidumbres Combinadas
with tab2:
    st.header("Calculadora de Incertidumbres Combinadas")
    
    st.write("""
    Esta sección te permite calcular el valor de una función y su incertidumbre combinada 
    a partir de los valores e incertidumbres de sus variables. La calculadora utiliza 
    la fórmula de propagación de incertidumbres:
    """)
    
    st.latex(r"u_c = \sqrt{\sum_{i=1}^{n} \left(\frac{\partial f}{\partial x_i} \cdot u_{x_i}\right)^2}")
    
    # Configuración inicial de la calculadora
    col1, col2 = st.columns(2)
    with col1:
        num_variables = st.number_input("Número de variables", min_value=1, max_value=10, value=2, step=1, key="ic_num_vars")
    with col2:
        num_constantes = st.number_input("Número de constantes", min_value=0, max_value=10, value=1, step=1, key="ic_num_const")
    
    # Contenedores para almacenar información
    variables_info = {}
    constantes_info = {}
    
    # Input para variables
    st.subheader("Variables")
    
    # Crear dos columnas para organizar mejor los inputs
    cols = st.columns(2)
    
    for i in range(num_variables):
        col_idx = i % 2  # Alternar entre columnas 0 y 1
        with cols[col_idx]:
            st.write(f"#### Variable {i+1}")
            var_symbol = st.text_input(f"Símbolo de la variable {i+1}", value=chr(120+i), key=f"var_symbol_{i}")  # Comienza con 'x', luego 'y', 'z', etc.
            var_value = st.number_input(f"Valor de {var_symbol}", value=1.0, format=f"%.{st.session_state.precision}f", key=f"var_value_{i}")
            var_uncertainty = st.number_input(f"Incertidumbre de {var_symbol}", value=0.1, min_value=0.0, format=f"%.{st.session_state.precision}f", key=f"var_uncert_{i}")
            
            variables_info[var_symbol] = {
                "valor": var_value,
                "incertidumbre": var_uncertainty
            }
    
    # Input para constantes (si hay alguna)
    if num_constantes > 0:
        st.subheader("Constantes")
        
        cols = st.columns(2)
        
        for i in range(num_constantes):
            col_idx = i % 2  # Alternar entre columnas 0 y 1
            with cols[col_idx]:
                st.write(f"#### Constante {i+1}")
                const_symbol = st.text_input(f"Símbolo de la constante {i+1}", value=chr(97+i), key=f"const_symbol_{i}")  # Comienza con 'a', luego 'b', 'c', etc.
                const_value = st.number_input(f"Valor de {const_symbol}", value=2.0, format=f"%.{st.session_state.precision}f", key=f"const_value_{i}")
                
                constantes_info[const_symbol] = {
                    "valor": const_value
                }
    
    # Input para la función
    st.subheader("Función")
    
    # Crear una guía para el usuario sobre cómo escribir la función
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
    
    # Mostrar los símbolos disponibles para ayudar al usuario
    available_symbols = ", ".join(list(variables_info.keys()) + list(constantes_info.keys()))
    st.write(f"Símbolos disponibles: {available_symbols}")
    
    # Input para la función
    function_str = st.text_input("Función f(variables, constantes)", value="x**2 + a*y", key="function_input")
    st.subheader("Opción avanzada: escribir función en LaTeX")
    latex_str = st.text_area(
        "Función en LaTeX (ej: x^2 + a y, sin(x), \\pi*x)", 
        value="", 
        help="Si tu función es complicada o usa constantes especiales como \\pi, escribe aquí."
    )

    # Botón para calcular
    calcular_clicked = st.button("Calcular valor e incertidumbre")
    
    if calcular_clicked and function_str:
        try:
            # Crear símbolos sympy para cada variable
            symbols = {}
            for var in variables_info:
                symbols[var] = sp.symbols(var)
            for const in constantes_info:
                symbols[const] = sp.symbols(const)
            
            # Convertir la cadena de función a una expresión sympy
            if latex_str:
                try:
                    func_expr = parse_latex(latex_str)
                    st.write("Expresión convertida desde LaTeX:")
                    st.latex(sp.latex(func_expr))
                except Exception as e:
                    st.error(f"No se pudo convertir LaTeX a función: {e}")
                    func_expr = sp.sympify(function_str)  # fallback a Python
            else:
                func_expr = sp.sympify(function_str)
            
            # Mostrar la expresión en formato LaTeX
            st.write("### Función a evaluar:")
            st.latex(sp.latex(func_expr))
            
            # Calcular derivadas parciales para cada variable (no constantes)
            st.write("### Derivadas parciales:")
            derivatives = {}
            
            for var_symbol in variables_info:
                derivative = sp.diff(func_expr, symbols[var_symbol])
                derivatives[var_symbol] = derivative
                st.write(f"Derivada parcial respecto a {var_symbol}:")
                st.latex(r"\frac{\partial f}{\partial " + var_symbol + "} = " + sp.latex(derivative))
            
            # Crear un diccionario de valores para evaluación
            values_dict = {}
            for var in variables_info:
                values_dict[var] = variables_info[var]["valor"]
            for const in constantes_info:
                values_dict[const] = constantes_info[const]["valor"]
            
            # Evaluar la función con los valores dados
            func_value = float(func_expr.evalf(subs=values_dict))
            
            # Evaluar las derivadas parciales
            derivative_values = {}
            for var_symbol, derivative in derivatives.items():
                derivative_values[var_symbol] = float(derivative.evalf(subs=values_dict))
            
            # Calcular la incertidumbre combinada
            squared_terms = []
            for var_symbol in variables_info:
                derivative_value = derivative_values[var_symbol]
                uncertainty = variables_info[var_symbol]["incertidumbre"]
                term = (derivative_value * uncertainty)**2
                squared_terms.append(term)
            
            combined_uncertainty = np.sqrt(np.sum(squared_terms))
            
            # Mostrar resultados en forma de tabla
            results_data = []
            
            # Primero añadir el valor de la función
            results_data.append({
                "Descripción": "Valor de la función",
                "Valor": f"{func_value:.{st.session_state.precision}f}"
            })
            
            # Añadir la incertidumbre combinada
            results_data.append({
                "Descripción": "Incertidumbre combinada",
                "Valor": f"{combined_uncertainty:.{st.session_state.precision}f}"
            })
            
            # Añadir el resultado final con incertidumbre
            results_data.append({
                "Descripción": "Resultado final",
                "Valor": f"{func_value:.{st.session_state.precision}f} ± {combined_uncertainty:.{st.session_state.precision}f}"
            })
            
            # Añadir la incertidumbre relativa en porcentaje
            if func_value != 0:
                rel_uncertainty = (combined_uncertainty / abs(func_value)) * 100
                results_data.append({
                    "Descripción": "Incertidumbre relativa",
                    "Valor": f"{rel_uncertainty:.{st.session_state.precision}f}%"
                })
            
            # Mostrar la tabla de resultados
            results_df = pd.DataFrame(results_data)
            st.write("### Resultados:")
            st.table(results_df)
            
            # Mostrar los términos individuales que contribuyen a la incertidumbre
            st.write("### Contribuciones a la incertidumbre:")
            contributions = []
            
            for var_symbol in variables_info:
                derivative_value = derivative_values[var_symbol]
                uncertainty = variables_info[var_symbol]["incertidumbre"]
                term = (derivative_value * uncertainty)**2
                percent_contribution = (term / np.sum(squared_terms)) * 100
                
                contributions.append({
                    "Variable": var_symbol,
                    "Derivada parcial": f"{derivative_value:.{st.session_state.precision}f}",
                    "Incertidumbre": f"{uncertainty:.{st.session_state.precision}f}",
                    "Término (∂f/∂x·u)²": f"{term:.{st.session_state.precision}f}",
                    "Contribución (%)": f"{percent_contribution:.2f}%"
                })
            
            contrib_df = pd.DataFrame(contributions)
            st.table(contrib_df)
            
            # Visualizar las contribuciones a la incertidumbre en un gráfico de barras
            if len(variables_info) > 1:  # Solo mostrar el gráfico si hay más de una variable
                st.write("### Visualización de contribuciones a la incertidumbre:")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                variables = contrib_df["Variable"]
                contributions_pct = [float(x.strip('%')) for x in contrib_df["Contribución (%)"].values]
                
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(variables)))
                bars = ax.bar(variables, contributions_pct, color=colors)
                
                # Añadir etiquetas
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{height:.2f}%', ha='center', va='bottom')
                
                ax.set_xlabel('Variables')
                ax.set_ylabel('Contribución a la incertidumbre total (%)')
                ax.set_title('Contribución de cada variable a la incertidumbre combinada')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error al calcular: {e}")
            st.write("Verifica que la función esté correctamente escrita y que uses los símbolos definidos.")
with tab3:
    st.header("Gráficas automáticas con incertidumbres")

    st.write("Puedes introducir tus datos manualmente o subir un CSV con tus mediciones. También puedes superponer varias gráficas para comparar.")

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    modo = st.radio("Modo de entrada de datos:", ["Manual", "CSV"])

    # --- MODO MANUAL ---
    if modo == "Manual":
        n_puntos = st.number_input("Número de puntos", min_value=2, value=4, step=1)

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

        if st.button("Generar gráfica manual"):
            xs = np.array([d[0] for d in datos])
            dxs = np.array([d[1] for d in datos])
            ys = np.array([d[2] for d in datos])
            dys = np.array([d[3] for d in datos])

            fig, ax = plt.subplots()
            ax.errorbar(xs, ys, xerr=dxs, yerr=dys, fmt='o-', capsize=4,
                        ecolor='black', color='tab:blue', label='Datos experimentales')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Gráfico con barras de error")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.info("La línea une los puntos medidos y los palitos representan las incertidumbres.")

    # --- MODO CSV ---
    elif modo == "CSV":
        uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa del CSV:")
            st.dataframe(df.head())

            n_graficas = st.number_input("Número de gráficas a superponer", min_value=1, value=1, step=1)

            # Guarda las selecciones en el estado de la sesión para que no se pierdan
            if "graficas" not in st.session_state:
                st.session_state.graficas = []

            colores = plt.cm.tab10.colors

            for i in range(int(n_graficas)):
                st.subheader(f"Gráfica {i+1}")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_col = st.selectbox(f"X (Gráfica {i+1})", df.columns, key=f"x_col_{i}")
                with col2:
                    dx_col = st.selectbox(f"ΔX (Gráfica {i+1})", [None] + list(df.columns), key=f"dx_col_{i}")
                with col3:
                    y_col = st.selectbox(f"Y (Gráfica {i+1})", df.columns, key=f"y_col_{i}")
                with col4:
                    dy_col = st.selectbox(f"ΔY (Gráfica {i+1})", [None] + list(df.columns), key=f"dy_col_{i}")

                if st.button(f"Añadir gráfica {i+1}"):
                    xs = df[x_col].values
                    dxs = df[dx_col].values if dx_col is not None else np.zeros_like(xs)
                    ys = df[y_col].values
                    dys = df[dy_col].values if dy_col is not None else np.zeros_like(ys)

                    st.session_state.graficas.append({
                        "xs": xs,
                        "dxs": dxs,
                        "ys": ys,
                        "dys": dys,
                        "color": colores[i % len(colores)],
                        "label": f"Gráfica {i+1}"
                    })
                    st.success(f"Gráfica {i+1} añadida correctamente ✅")

            if st.button("Mostrar todas las gráficas"):
                if len(st.session_state.graficas) == 0:
                    st.warning("Primero añade al menos una gráfica.")
                else:
                    fig, ax = plt.subplots()
                    for g in st.session_state.graficas:
                        ax.errorbar(
                            g["xs"], g["ys"], xerr=g["dxs"], yerr=g["dys"],
                            fmt='o-', capsize=4, ecolor='black',
                            color=g["color"], label=g["label"]
                        )
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_title("Gráfico con múltiples conjuntos de datos")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    st.info("Cada color representa un conjunto de datos distinto con sus incertidumbres.")
