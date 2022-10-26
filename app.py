import cv2
import gradio as gr
import numpy as np
import phasepack.phasecong as pc


def _gradient_magnitude(img: np.ndarray, img_depth: int):
    """
    Calculate gradient magnitude based on Scharr operator.
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)


def _similarity_measure(x: np.array, y: np.array, constant: float):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator


def fsim(
        img1: np.ndarray, img2: np.ndarray, T1: float = 0.85, T2: float = 160):
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)
    Args:
        img1 -- original image
        img2 -- image to be compared 
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """
    alpha = (
        beta
    ) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(img1.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(
            img1[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )
        pc2_2dim = pc(
            img2[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros(
            (img1.shape[0], img1.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros(
            (img2.shape[0], img2.shape[1]), dtype=np.float64
        )
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(img1[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(img2[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


def similarity(img1, img2):
    if img1.shape == img2.shape:
        score = fsim(img1, img2)
        if score*100 > 27:
            return "The images are similar"
        else:
            return "The images are not similar"


demo = gr.Blocks()  # Create a gradio block

with demo:
    gr.Markdown("# Image Similarity using Feature Based Similarity Index")
    with gr.Tabs():
        with gr.TabItem("Examples"):  # If the user wants to use the examples
            with gr.Row():
                rad1 = gr.components.Radio(
                    ['Image 1', 'Image 2'], label='Select Image and wait till it appears!')  # Radio button to select the article
                # Textbox to show the article
                img1 = gr.Image(label="Image 1", shape=(300, 300))
                rad2 = gr.components.Radio(
                    ['Image 3', 'Image 4'], label='Select Image and wait till it appears!')  # Radio button to select the article
                img2 = gr.Image(label="Image 2", shape=(300, 300))
            submit1 = gr.Button('Submit')
        with gr.TabItem("Do it yourself!"):  # If the user wants to enter their own text
            with gr.Row():
                img3 = gr.Image(label="Image 1", shape=(300, 300))
                img4 = gr.Image(label="Image 2", shape=(300, 300))
            submit2 = gr.Button('Submit')

        def action1(choice):  # Function to show the article when the user selects the article
            if choice == 'Image 1':
                return 'images/shoe.jpg'
            elif choice == 'Image 2':
                return 'images/printer1.jpg'
            elif choice == 'Image 3':
                return 'images/shoe1.jpg'
            elif choice == 'Image 4':
                return 'images/printer2.jpg'

        # Change the article when the user selects the article
        rad1.change(action1, rad1, img1)
        rad2.change(action1, rad2, img2)

        # Output for the Highlighted text
        op = gr.components.Textbox(label="Similarity", lines=1)

        gr.Markdown(
            "### Made with ❤️ by Arsh using TrueFoundry's Gradio Deployment")
        gr.Markdown(
            "### [Github Repo](https://github.com/d4rk-lucif3r/Feature-based-Similarity-Index)")
        gr.Markdown(
            '### [Blog]()')

        def fn(img1, img2):  # Main function
            result = similarity(img1, img2)
            return result

        submit1.click(fn=fn, inputs=[img1, img2], outputs=[
                      op])  # Submit button for the examples
        # Submit button for the user input
        submit2.click(fn=fn, inputs=[img3, img4], outputs=[op])

demo.launch(server_port=8080, server_name='0.0.0.0')  # Launch the gradio block
