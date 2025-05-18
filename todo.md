# Team focus check in 3/6/24


options:
 - keep cloud but not do gans
   - supervised ml that sees through clouds sent2 and landsat -> create mask

   
default task:
 - ground truth
   - use another model for semantic segmentation
   - Look up models(huggingface)
   - https://huggingface.co/models?pipeline_tag=image-segmentation&sort=trending
   - if model is vision transformer(ViT) -> model architecture is diff from encoder-decoder architecture
      - make sure to use pre-trained weights 
      - adjust final layer to output matrices of correct dimension
        - view resnet_transfer.py

dependent on hw3..
3rd option:
 - look at results of hw3 and run explainable ai output for confidence levels
 - https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Semantic%20Segmentation.html
 
4th option:
 - do bold goal with interactive map
    - UI with streamlit and grad.io?
    - https://huggingface.co/spaces/gradio/image_segmentation
    = https://huggingface.co/spaces/ajcdp/Image-Segmentation-Gradio/tree/main 


