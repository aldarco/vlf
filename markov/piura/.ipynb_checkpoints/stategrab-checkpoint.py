import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import CustomJS, TapTool, Range1d, LinearColorMapper
from bokeh.layouts import column
from datetime import datetime, timedelta

# --- 1. Preparación de los Datos ---
# Simular tu serie de tiempo con gaps y un índice datetime
start_date = datetime(2025, 3, 7, 12, 0, 0)
end_date = datetime(2025, 5, 26, 22, 0, 0)
time_range = pd.date_range(start=start_date, end=end_date, freq='1min')

# Filtrar para el rango horario 12:00-22:00
df = pd.DataFrame(index=time_range)
df['value'] = np.random.randn(len(df))

# Aplicar el filtro horario
df = df[(df.index.hour >= 12) & (df.index.hour <= 22)]

# Columna para los estados (valor por defecto: 1)
df['state'] = 1

# Convertir el índice a milisegundos desde la época para Bokeh CustomJS
df['time_ms'] = df.index.astype(np.int64) // 10**6

# --- 2. Configuración Inicial del Gráfico ---
# Seleccionar el primer día para mostrar
current_day = df.index.min().normalize()

def get_data_for_day(date_obj):
    """Filtra el DataFrame para un día específico."""
    start_of_day = date_obj.normalize() + timedelta(hours=12) # Empezar desde las 12:00
    end_of_day = date_obj.normalize() + timedelta(hours=22, minutes=59, seconds=59) # Terminar a las 22:00
    return df[(df.index >= start_of_day) & (df.index <= end_of_day)]

# Crear el ColumnDataSource inicial
initial_data_for_day = get_data_for_day(current_day)
source = ColumnDataSource(data={
    'time_ms': initial_data_for_day['time_ms'].tolist(),
    'value': initial_data_for_day['value'].tolist(),
    'state': initial_data_for_day['state'].tolist(),
    'index_df': initial_data_for_day.index.astype(str).tolist() # Convertir a string para JSON
})

# Crear la figura
p = figure(
    x_axis_type="datetime",
    title=f"Serie de Tiempo: {current_day.strftime('%Y-%m-%d')}",
    height=400,
    width=900,
    tools="pan,wheel_zoom,box_zoom,reset,save" # TapTool se añade con p.add_tools
)

# Línea principal de los datos
p.line(x='time_ms', y='value', source=source, line_width=2, color='blue', legend_label="Valor")

# Configurar el mapeo de colores para los estados
color_mapper = LinearColorMapper(palette=['blue', 'green', 'red'], low=1, high=3)

# Círculos para los puntos de datos (donde se harán los clics)
circles_source = ColumnDataSource(data={
    'time_ms': initial_data_for_day['time_ms'].tolist(),
    'value': initial_data_for_day['value'].tolist(),
    'state': initial_data_for_day['state'].tolist()
})

# CORRECTED: Using p.scatter() instead of p.circle() for future compatibility
circles = p.scatter(x='time_ms', y='value', source=circles_source, size=8, alpha=0.6, marker='circle', # Explicitly specify marker
                   fill_color={'field': 'state', 'transform': color_mapper},
                   line_color='black', # Optional: Add a black border for clarity
                   legend_label="Puntos")

# Añadir un TapTool (aunque la lógica de clic se maneja con CustomJS)
p.add_tools(TapTool())

# --- 3. Manejo Interactivo con CustomJS ---

# JavaScript para manejar los clics del ratón (izquierdo y derecho)
# MOVED THIS DEFINITION HERE, BEFORE IT'S CALLED
js_callback_click = CustomJS(args=dict(source=source, p=p, df_full=df.reset_index().astype(str).to_json(orient='records'),
                                        circles_source=circles_source, color_mapper=color_mapper), code="""
    // Convertir el DataFrame completo de JS a Pandas-like para búsquedas eficientes
    const df_raw = JSON.parse(df_full);
    const df_map = new Map(); // Mapa para búsquedas rápidas por índice datetime string
    df_raw.forEach(row => {
        df_map.set(row.index, row);
    });

    p.canvas_view.el.addEventListener('mousedown', function(event) {
        let new_state = 0;
        if (event.button === 0) { // Clic izquierdo
            new_state = 2;
        } else if (event.button === 2) { // Clic derecho
            new_state = 3;
        }

        if (new_state !== 0) {
            // Obtener las coordenadas del clic en el espacio de datos
            const x_coord = p.x_range.inverse(event.offsetX);
            const y_coord = p.y_range.inverse(event.offsetY);

            // Encontrar el punto más cercano en el ColumnDataSource
            let min_dist = Infinity;
            let closest_index = -1;
            const data_x = source.data['time_ms'];
            const data_y = source.data['value'];

            for (let i = 0; i < data_x.length; i++) {
                // Calcular distancia euclidiana (aproximada para fines de selección)
                const dist_x = Math.abs(x_coord - data_x[i]);
                const dist_y = Math.abs(y_coord - data_y[i]);
                const dist = Math.sqrt(Math.pow(dist_x, 2) + Math.pow(dist_y, 2)); // Distancia euclidiana
                
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_index = i;
                }
            }
            
            // Un umbral para asegurarse de que el clic esté "cerca" de un punto
            const tolerance_x_pixels = 10; // 10 píxeles de tolerancia en X
            const tolerance_y_pixels = 10; // 10 píxeles de tolerancia en Y

            // Convertir tolerancia de píxeles a unidades de datos
            const x_pixel_to_data_ratio = (p.x_range.end - p.x_range.start) / p.canvas_view.frame.width;
            const y_pixel_to_data_ratio = (p.y_range.end - p.y_range.start) / p.canvas_view.frame.height;

            const tolerance_x_data = tolerance_x_pixels * x_pixel_to_data_ratio;
            const tolerance_y_data = tolerance_y_pixels * y_pixel_to_data_ratio;

            if (closest_index !== -1 &&
                Math.abs(x_coord - data_x[closest_index]) < tolerance_x_data &&
                Math.abs(y_coord - data_y[closest_index]) < tolerance_y_data) {
                
                // Actualizar el estado en el ColumnDataSource visible
                source.data['state'][closest_index] = new_state;
                circles_source.data['state'][closest_index] = new_state;

                // Encontrar y actualizar el estado en el DataFrame 'global' (df_map)
                // Esto es crucial para que los cambios persistan al cambiar de día
                const original_index_str = source.data['index_df'][closest_index];
                if (df_map.has(original_index_str)) {
                    df_map.get(original_index_str).state = new_state;
                    console.log(`Updated original df at ${original_index_str} to state ${new_state}`);
                }

                source.change.emit(); // Notificar a Bokeh que los datos han cambiado
                circles_source.change.emit(); // Notificar a Bokeh que los datos de los círculos han cambiado
            }
        }
    });

    // Deshabilitar el menú contextual del clic derecho para que Bokeh pueda capturarlo
    p.canvas_view.el.addEventListener('contextmenu', function(event) {
        event.preventDefault();
    });
""")


# JavaScript para manejar el estado del día actual y la navegación
js_callback_navigate = CustomJS(args=dict(source=source, p=p, df_full=df.reset_index().astype(str).to_json(orient='records'),
                                           circles_source=circles_source, color_mapper=color_mapper), code="""
    // Convertir el DataFrame completo de JS a Pandas-like para búsquedas eficientes
    const df_raw = JSON.parse(df_full);
    const df_map = new Map(); // Mapa para búsquedas rápidas por índice datetime string
    df_raw.forEach(row => {
        df_map.set(row.index, row);
    });

    let currentDayMs = p.x_range.start; // Usamos el inicio del eje X como el día actual en ms

    // Event listener para manejar la navegación con la flecha derecha
    document.addEventListener('keydown', function(event) {
        if (event.key === 'ArrowRight') {
            // Calcular el inicio del día actual (sin las horas/minutos)
            const currentDayStart = new Date(currentDayMs);
            currentDayStart.setHours(0, 0, 0, 0);
            
            // Avanzar un día completo desde el inicio del día actual
            const nextDayStartMs = currentDayStart.getTime() + (24 * 60 * 60 * 1000);
            const nextDay = new Date(nextDayStartMs);
            
            // Filtrar los datos para el nuevo día
            const new_data = {
                time_ms: [],
                value: [],
                state: [],
                index_df: []
            };

            // Iterar sobre el DataFrame completo para encontrar datos del día
            df_raw.forEach(row => {
                const rowDate = new Date(row.index); // Convertir el string de fecha a objeto Date
                
                // Comprobar si la fecha coincide con el día siguiente y está dentro del rango horario
                if (rowDate.toDateString() === nextDay.toDateString() &&
                    rowDate.getHours() >= 12 && rowDate.getHours() <= 22) {
                    new_data.time_ms.push(row.time_ms);
                    new_data.value.push(row.value);
                    new_data.state.push(row.state);
                    new_data.index_df.push(row.index);
                }
            });

            if (new_data.time_ms.length > 0) {
                source.data = new_data;
                circles_source.data = { // Actualizar también la fuente de los círculos
                    time_ms: new_data.time_ms,
                    value: new_data.value,
                    state: new_data.state
                };

                // Actualizar el día actual en JavaScript
                currentDayMs = nextDay.getTime(); // Actualizar para el siguiente cálculo

                // Actualizar título del gráfico
                p.title.text = "Serie de Tiempo: " + nextDay.toLocaleDateString();
                
                // Ajustar el rango del eje X para el nuevo día
                const startOfDayRange = new Date(nextDay);
                