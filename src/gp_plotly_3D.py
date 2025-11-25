import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data.iloc[:, 2:5].values, data['type'].values


def get_clf(x_data, y_data):
    gpc = GaussianProcessClassifier()
    clf = make_pipeline(StandardScaler(), gpc)
    clf.fit(X=x_data, y=y_data)
    # gpc_model = clf.named_steps['gaussianprocessclassifier']
    # print("Optimized kernel:", gpc_model.kernel_)
    return clf


def main():
    csv_path = r".\data\data_sorted_by_No_in_article.csv"
    value, type = read_data(csv_path)
    clf = get_clf(value, type)
    label = clf.predict(value)
    print(label)

    # Define 3D space range and sampling density
    X_MIN, X_MAX = -70, 15
    Y_MIN, Y_MAX = 10, 70
    Z_MIN, Z_MAX = -70, 160
    SAMPLE_DEN = 0.25  # 0.5 high definition
    
    # Define colors for different classes
    mesh_color = ['red', 'gold', 'blue', 'green']
    color_maps = ['reds', 'oranges', 'blues', 'greens']
    # Create color mapping (class name to color)
    class_names = np.unique(type)
    color_map = {class_name: mesh_color[i] for i, class_name in enumerate(class_names)}
    
    # Generate regular grid
    xx, yy, zz = np.meshgrid(
        np.linspace(X_MIN, X_MAX, int(np.round(SAMPLE_DEN * (X_MAX - X_MIN)))),
        np.linspace(Y_MIN, Y_MAX, int(np.round(SAMPLE_DEN * (Y_MAX - Y_MIN)))),
        np.linspace(Z_MIN, Z_MAX, int(np.round(SAMPLE_DEN * (Z_MAX - Z_MIN))))
    )
    x_len, y_len, z_len = xx.shape
    print(x_len*y_len*z_len)
    
    # Predict class probabilities for grid points
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    S = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])  # [np.arange(len(Z)), Z.astype(int)]

    # Z = Z.reshape(xx.shape)
    S = S.reshape((x_len, y_len, z_len, 4))
    # print(Z)

    # Create figure object
    fig = go.Figure()

    for i in [0, 2, 3, 1]:
        # Add volume plot
        fig.add_trace(go.Volume(
            x=xx.flatten(), 
            y=yy.flatten(), 
            z=zz.flatten(),
            value=S[:, :, :, i].flatten(),
            isomin=0.35,
            isomax=0.6,
            opacity=1,
            surface_count=50,
            colorscale=color_maps[i], 
            lighting=dict(
            ambient=0.99,    # Ambient light intensity (1.0 means fully uniform lighting)
            diffuse=0,    # Diffuse light intensity (0 disables)
            specular=0.0,   # Specular reflection intensity (0 disables)
            roughness=1.0   # Surface roughness (1.0 means completely rough, no reflection)
            ),
            showlegend=False, 
            showscale=False
        ))
    
    # Add original data points (colored by predicted class)
    for class_name in [0, 2, 3, 1]:
        # Get data points for current class
        class_mask = (label == class_name)
        value_class = value[class_mask]
        
        # Add scatter plot
        fig.add_trace(go.Scatter3d(
            x=value_class[:, 0],
            y=value_class[:, 1],
            z=value_class[:, 2],
            mode='markers',
            marker=dict(
                size=5,  # 0.75
                color=color_map[class_name],  # Use color corresponding to class
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
                showticklabels=True, 
                gridcolor='black', 
                showbackground=False, 
                title='Attachability', 
                title_font=dict(size=18, family='Arial', color='black'),  # Increase title font size
                tickfont=dict(size=14, family='Arial', color='black'),    # Increase tick font size
                linewidth=4,           # Thicken axis lines
                gridwidth=2,           # Thicken grid lines
                linecolor='black'      # Set axis line color
            ),
            yaxis=dict(
                showticklabels=True, 
                gridcolor='black', 
                showbackground=False, 
                title='Controllability', 
                title_font=dict(size=18, family='Arial', color='black'),  # Increase title font size
                tickfont=dict(size=14, family='Arial', color='black'),    # Increase tick font size
                linewidth=4,           # Thicken axis lines
                gridwidth=2,           # Thicken grid lines
                linecolor='black'      # Set axis line color
            ),
            zaxis=dict(
                showticklabels=True, 
                gridcolor='black', 
                showbackground=False, 
                title='Detachability', 
                title_font=dict(size=18, family='Arial', color='black'),  # Increase title font size
                tickfont=dict(size=14, family='Arial', color='black'),    # Increase tick font size
                linewidth=4,           # Thicken axis lines
                gridwidth=2,           # Thicken grid lines
                linecolor='black'      # Set axis line color
            ),
            camera=dict(
                projection=dict(type='orthographic'),
                eye=dict(x=-0.7752781775459654, y=1.4528183583539271, z=1.2609368632102829),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.85)
        ),
        # title='3D Classification with Data Points',
        width=1600,  #6400 Increase width for higher resolution
        height=1200, #4800 Increase height for higher resolution
        title="",
        showlegend=False  # Globally disable legend
    )

    # # Export as high-quality PNG
    # pio.kaleido.scope.default_format = "png"
    # fig.write_image(r".\output3\all.png", 
    #                scale=8,  # 4x supersampling for higher resolution
    #                engine="kaleido")
    
    # print("High-quality PNG image saved as '3d_classification_no_legend.png'")

    fig.write_html("3d_volume_classification.html", auto_open=True)
    # fig.show()


    """
    Find viewing angle: Enter the following code in the HTML console to get camera parameters
    var plotDiv = document.querySelector('.plotly-graph-div');
    var camera = plotDiv.layout.scene.camera;
    console.log(JSON.stringify(camera, null, 2));
    """


if __name__ == '__main__':
    main()
    
