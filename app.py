from flask import Flask, render_template, request
import os

app = Flask(__name__)
DATASET_FOLDER = 'dataset_daun-kentang'
app.config['DATASET_FOLDER'] = DATASET_FOLDER

# Ambil daftar gambar dan kategorinya
def get_image_list():
    categories = {}
    for category in os.listdir(DATASET_FOLDER):
        category_path = os.path.join(DATASET_FOLDER, category)
        if os.path.isdir(category_path):
            categories[category] = [f for f in os.listdir(category_path) if f.endswith(('jpg', 'png', 'jpeg'))]
    return categories

@app.route('/', methods=['GET', 'POST'])
def index():
    categories = get_image_list()
    selected_image = None
    selected_category = None
    
    if request.method == 'POST':
        selected_image = request.form.get('selected_image')
        for category, images in categories.items():
            if selected_image in images:
                selected_category = category
                break
    
    return render_template('index.html', categories=categories, selected_image=selected_image, selected_category=selected_category)

if __name__ == '__main__':
    app.run(debug=True)
