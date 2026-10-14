import gradio as gr
from src.predict import load_model, predict_image, GRADE_NAMES

model, device = load_model()

def classify(image):
    if image is None:
        return {}, 'Please upload a retinal fundus image.'
    result = predict_image(image, model, device)
    scores = {
        label: result['class_scores_percent'][label] / 100
        for label in GRADE_NAMES
    }
    message = (
        f"**Predicted grade:** {result['predicted_label']}\n\n"
        f"**Softmax score:** {result['softmax_score_percent']}%\n\n"
        'For research and educational use only. Not a medical diagnosis.'
    )
    return scores, message

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type='filepath', label='Upload retinal fundus image'),
    outputs=[
        gr.Label(num_top_classes=5, label='Model class scores'),
        gr.Markdown(label='Prediction'),
    ],
    title='Diabetic Retinopathy Classification',
    description='Experimental EfficientNet-B0 model trained on APTOS 2019. Not for clinical use.',
)

if __name__ == '__main__':
    demo.launch(share=True)
