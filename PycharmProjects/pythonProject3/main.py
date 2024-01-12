# You can run the app using the command 'streamlit run main.py'  in terminal.
# Please make sure to adjust the 'model_path' variable according to the location where your models are saved.

import streamlit as st
from keras.models import load_model
from keras.utils import custom_object_scope
from PIL import Image
from util import classify
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
import tempfile
import zipfile
import os
import tensorflow as tf
import torch
from util import count_sparse_wgt_by_layer,CustomNet ,count_sparse_wgt_by_filter,count_sparse_wgt,count_sparse_wgt_by_channel,print_sparse_weights

def load_and_display_model(model_choice):
    if model_choice == 'Unpruned Model':
        model_path = r'C:\Users\reyha\PycharmProjects\pythonProject2\unpruned_model.h5'
    elif model_choice == 'Weight Pruning Model':
        model_path = r'C:\Users\reyha\PycharmProjects\pythonProject2\w_pruning_model.h5'
    elif model_choice == 'Filter Pruning Model':
        model_path = r"C:\Users\reyha\PycharmProjects\pythonProject2\filter_pruning_model.h5"

    if model_choice == 'Unpruned Model' or model_choice == 'Compare Models':
        model = load_model(model_path)
    else:
        with custom_object_scope({'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude}):
            model = load_model(model_path)

    return model


def load_labels(file_path='labels.txt'):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def display_model_summary(model):
    st.text("Model Summary:")
    model.summary(print_fn=st.text)

def get_model_size_info(model, model_choice):
    try:
        # Save the model to a temporary file
        _, temp_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model, temp_keras_file, include_optimizer=False)

        # Zip the temporary file for compression
        _, temp_zip_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(temp_zip_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(temp_keras_file)

        model_size_before = os.path.getsize(temp_keras_file) / float(2**20)
        model_size_after = os.path.getsize(temp_zip_file) / float(2**20)

        st.text(f"Size of the {model_choice} model before compression: {model_size_before:.2f} Mb")
        st.text(f"Size of the {model_choice} model after compression: {model_size_after:.2f} Mb")

        return temp_keras_file, temp_zip_file

    except IOError as e:
        st.error(f"Error: {e}")
        return None, None

def main():
    st.title('Compare different NN model structure for CIFAR10 dataset')

    model_choices = ['Unpruned Model', 'Weight Pruning Model', 'Compare Models', 'Filter pruning Model']
    model_choice = st.selectbox('Select Model:', model_choices)

    compare_models = model_choice == 'Compare Models'
    filter_pruning = model_choice == 'Filter pruning Model'

    if compare_models:
        st.text("Model Size Comparison:")
        _, zip_unpruned = get_model_size_info(load_model('unpruned_model.h5'), 'Unpruned Model')
        _, zip_pruned = get_model_size_info(load_model('w_pruning_model.h5'), 'Weight Pruning Model')

    elif filter_pruning:
        model_path = r"C:\Users\reyha\PycharmProjects\pythonProject2\filter_pruning_model.h5"
        class_names = load_labels('labels.txt')
        st.title("Show me")

        # Ask the user which sparse weight information to display
        sparse_weight_option = st.radio("Select Sparse Weight Information:",
                                        ["Sparse Weight by Layer", "Sparse Weight by Filter"])

        if sparse_weight_option == "Sparse Weight by Layer":
            # Load your model and state_dict
            model = CustomNet(num_classes=len(class_names))
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)

            # Display sparse weight information by layer
            st.subheader("Sparse Weight Information by Layer")

            wgt_cnts, sparse_wgt_cnts = count_sparse_wgt_by_layer(model, threshold=1e-5)

            for idx in range(len(wgt_cnts)):
                layer_name = wgt_cnts[idx][0]
                wgt_cnt = wgt_cnts[idx][1]
                sparse_wgt_cnt = sparse_wgt_cnts[idx][1]
                sparsity_ratio = sparse_wgt_cnt / wgt_cnt

                st.write(f"Layer: {layer_name}, Sparse Weight: {sparsity_ratio:.3f} ({sparse_wgt_cnt}/{wgt_cnt})")



        elif sparse_weight_option == "Sparse Weight by Filter":

            # Load your model and state_dict
            model = CustomNet(num_classes=len(class_names))
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)

            # Display sparse weight information by filter
            st.subheader("Sparse Weight Information by Filter")
            sparse_wgt_cnts = count_sparse_wgt_by_filter(model, threshold=1e-5)
            for idx in range(len(sparse_wgt_cnts)):
                layer_name = sparse_wgt_cnts[idx][0]
                wgts_filters = sparse_wgt_cnts[idx][1]

                st.write(f"Layer: {layer_name}, Sparse Weight: ({wgts_filters})")

    else:
        model = load_and_display_model(model_choice)
        class_names = load_labels('labels.txt')

        action_choice = st.radio('Select an action:', ['View Model Summary', 'Upload Photo for Classification'])

        if action_choice == 'View Model Summary':
            display_model_summary(model)
        elif action_choice == 'Upload Photo for Classification':
            # Upload and classify image
            file = st.file_uploader('Please upload an image:', type=['jpeg', 'jpg', 'png'])
            if file is not None:
                image = Image.open(file).convert('RGB')
                image_resized = image.resize((32, 32))
                st.image(image_resized, use_column_width=True)
                class_name, conf_score = classify(image_resized, model, class_names)
                st.write("## {}".format(class_name))
                st.write("### Score: {}%".format(int(conf_score * 1000) / 10))


if __name__ == "__main__":
    main()
