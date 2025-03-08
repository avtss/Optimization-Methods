import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc


def func(x, y):
    #return 2*x**2 + 2*x*y + y**2
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
def gradient(x, y, h=1e-5):
    df_dx = (func(x + h, y) - func(x - h, y)) / (2 * h)
    df_dy = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

def gradient_descent(x0, y0, step=0.1, epsilon=1e-6, epsilon1=1e-6, epsilon2=1e-6, max_iter=100):
    history = []
    current_point = np.array([x0, y0])
    
    for i in range(max_iter):   
        grad = gradient(*current_point)
        current_value = func(*current_point)
                
        grad_norm = np.linalg.norm(grad)
        
        history.append({
            'iteration': i+1,
            'x': current_point[0],
            'y': current_point[1],
            'f_value': current_value,
            'grad_norm': grad_norm
        })
        
        if grad_norm < epsilon1:
            return history, True, "Сошёлся (норма градиента меньше заданной точности)"
        
        old_point = current_point 
        current_point = current_point - step * grad
        modified_step = step

        while not(func(*current_point) - func(*old_point)) < 0:
            modified_step = modified_step / 2
            current_point = old_point - modified_step * grad

        if (np.linalg.norm(current_point-old_point) < epsilon2) and (np.linalg.norm(func(*current_point)-func(*old_point)) < epsilon2):
            return history, True, "Сошёлся (разница значений функции меньше заданной точности)" 
    
    return history, False, "Не сошёлся (достигнуто максимальное количество итераций)"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server

app.layout = dbc.Container([
    html.H1("Градиентный спуск с постоянным шагом", className='text-center mt-4 mb-4'),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Параметры", className="bg-primary text-white text-center"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([html.Label("X₀")], width=4),
                        dbc.Col([dbc.Input(id='x0-input', type='number', value=0)], width=8),
                    ], className="mb-2"),

                    dbc.Row([
                        dbc.Col([html.Label("Y₀")], width=4),
                        dbc.Col([dbc.Input(id='y0-input', type='number', value=0)], width=8),
                    ], className="mb-2"),

                    dbc.Row([
                        dbc.Col([html.Label("Шаг")], width=4),
                        dbc.Col([dbc.Input(id='lr-input', type='number', value=0.1, step=0.01)], width=8),
                    ], className="mb-2"),

                    dbc.Row([
                        dbc.Col([html.Label("Точность ε")], width=4),
                        dbc.Col([dbc.Input(id='epsilon-input', type='number', value=1e-4, step=1e-6)], width=8),
                    ], className="mb-2"),

                    dbc.Row([
                        dbc.Col([html.Label("Макс. итераций")], width=4),
                        dbc.Col([dbc.Input(id='max-iter-input', type='number', value=100)], width=8),
                    ], className="mb-3"),

                    dbc.Button("Запустить", id='run-button', color='success', className='w-100 mt-3')
                ])
            ], className="mb-4")
        ], md=4),

        dbc.Col([
            dcc.Graph(id='3d-plot', style={'height': '450px'}),
            html.Div(id='results-table', className='mt-3'),
        ], md=8),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Alert(id='final-result', color="dark", className='mt-3 text-center')
        ])
    ]),
], fluid=True)


@app.callback(
    [Output('3d-plot', 'figure'),
     Output('results-table', 'children'),
     Output('final-result', 'children'),
     Output('final-result', 'color')],
    [Input('run-button', 'n_clicks')],
    [State('x0-input', 'value'),
     State('y0-input', 'value'),
     State('lr-input', 'value'),
     State('epsilon-input', 'value'),
     State('max-iter-input', 'value')]
)
def update_plot_and_table(n_clicks, x0, y0, lr, epsilon, max_iter):
    if None in [x0, y0, lr, epsilon, max_iter]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"
    
    history, converged, status_message = gradient_descent(x0, y0, lr, epsilon, epsilon, epsilon, max_iter)
    
    if history:
        final = history[-1]
        result_message = f"""
        Финальная точка: ({round(final['x'], 4)}, {round(final['y'], 4)})  
        Значение функции: {round(final['f_value'], 4)}  
        Итераций выполнено: {final['iteration']}  
        Состояние: {status_message}
        """
        color = "success" if converged else "warning"
    else:
        result_message = "Не удалось выполнить оптимизацию"
        color = "danger"
    
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    trajectory_x = [point['x'] for point in history]
    trajectory_y = [point['y'] for point in history]
    trajectory_z = [point['f_value'] for point in history]
    
    fig = go.Figure(data=[
        go.Surface(x=X, y=Y, z=Z, colorscale='Plasma', opacity=0.8),
        go.Scatter3d(
            x=trajectory_x,
            y=trajectory_y,
            z=trajectory_z,
            mode='lines+markers',
            marker=dict(size=5, color='red'),
            line=dict(color='red', width=2)
        )
    ])
    
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='f(x, y)'),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    table = dash_table.DataTable(
        columns=[{'name': i, 'id': i} for i in history[0].keys()],
        data=history,
        page_size=10,
        style_table={'overflowX': 'auto'}
    )
    
    return fig, table, result_message, color

if __name__ == '__main__':
    app.run_server(debug=True)
