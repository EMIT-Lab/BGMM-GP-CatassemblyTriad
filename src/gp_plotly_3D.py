import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
# import plotly.io as pio


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data.iloc[:, 2:5].values, data['type'].values


def get_clf(x_data, y_data):
    gpc = GaussianProcessClassifier()
    clf = make_pipeline(StandardScaler(), gpc)
    clf.fit(X=x_data, y=y_data)
    # gpc_model = clf.named_steps['gaussianprocessclassifier']
    return clf


def main():
    csv_path = r".\data\data - forplot.csv"
    X, Y = read_data(csv_path)
    X[:, -1] = 0.1 * X[:, -1]
    clf = get_clf(X, Y)
    label = clf.predict(X)

    X_MIN, X_MAX = -85, 30
    Y_MIN, Y_MAX = -25, 92.5
    Z_MIN, Z_MAX = -33.5, 70
    SAMPLE_DEN = 0.25
    
    mesh_color = ['red', 'purple', 'gold', 'green', 'blue']
    class_names = np.unique(Y)
    
    color_map = {class_name: mesh_color[i] for i, class_name in enumerate(class_names)}
    
    xx, yy, zz = np.meshgrid(
        np.linspace(X_MIN, X_MAX, int(np.round(SAMPLE_DEN * (X_MAX - X_MIN)))),
        np.linspace(Y_MIN, Y_MAX, int(np.round(SAMPLE_DEN * (Y_MAX - Y_MIN)))),
        np.linspace(Z_MIN, Z_MAX, int(np.round(SAMPLE_DEN * (Z_MAX - Z_MIN))))
    )
    x_len, y_len, z_len = xx.shape
    print(f"The total number of sampling grid points: {x_len * y_len * z_len}")
    
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    S = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])  # [np.arange(len(Z)), Z.astype(int)]

    # Z = Z.reshape(xx.shape)
    S = S.reshape((x_len, y_len, z_len, 5))
    color_maps = ['reds', 'purples', 'oranges', 'greens', 'blues']
    
    fig = go.Figure()

    for i in [0, 1, 2, 3, 4]:
        fig.add_trace(go.Volume(
            x=xx.flatten(), 
            y=yy.flatten(), 
            z=zz.flatten(),
            value=S[:, :, :, i].flatten(),
            isomin=0.25, # 0.24,
            isomax=0.55,
            opacity=1,
            surface_count=25,
            colorscale=color_maps[i], 
            lighting=dict(
            ambient=0.99,   
            diffuse=0,
            specular=0.0,  
            roughness=1.0 
            ),
            showlegend=False, 
            showscale=False
        ))
    
    for class_name in [0, 1, 2, 3, 4]:
        class_mask = (label == class_name)
        X_class = X[class_mask]
        
        fig.add_trace(go.Scatter3d(
            x=X_class[:, 0],
            y=X_class[:, 1],
            z=X_class[:, 2],
            mode='markers',
            marker=dict(
                size=0.75,
                color=color_map[class_name],
                opacity=0.8
            ),
            showlegend=False, 
            name=f'{class_name} Points'
        ))
    
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='gray',
        scene=dict(
            bgcolor='white',
            xaxis=dict(
                showticklabels=False, 
                gridcolor='black',    
                showbackground=False,
                # title='',             
                title='Attachability',
            ),
            yaxis=dict(
                showticklabels=False,
                gridcolor='black',
                showbackground=False,
                # title='',
                title='Detachability', 
            ),
            zaxis=dict(
                showticklabels=False,
                gridcolor='black',
                showbackground=False,
                # title='',
                title='Controllability', 
            ),
            camera=dict(
                projection=dict(type='orthographic'),
                eye=dict(x=-0.6420744720489857, y=1.0094542414268326, z=1.8040625008043352),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=6400,
        height=4800,
        title="",
        showlegend=False
    )

    # fig.write_image("GP3D_plot.pdf", scale=8)
    fig.write_image(r'.\output\GP3D_plot.png', scale=8)
    fig.write_html(r'.\output\This_can_change_the_perspective.html', auto_open=False)

    # fig.show()


    """
    Adjust the perspective (on F12 console)

    var plotDiv = document.querySelector('.plotly-graph-div');
    var camera = plotDiv.layout.scene.camera;
    console.log(JSON.stringify(camera, null, 2));
    """


if __name__ == '__main__':
    main()