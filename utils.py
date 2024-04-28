import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
import seaborn as sns

def load_and_transform_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


def visualize_basic_attention(image_input, text_input, image_weights, text_weights, id2token):
    """
    Visualizes the attention weights for both image and text inputs.
    
    :param image_input: The original image tensor.
    :param text_input: The original text input IDs.
    :param image_weights: Attention weights for the image.
    :param text_weights: Attention weights for the text.
    :param id2token: A dictionary to convert token IDs back to strings for visualization.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    id2token = tokenizer.get_vocab()
    id2token = {v: k for k, v in id2token.items()}  # Invert the dictionary
    # Normalize weights for better visualization
    image_weights = image_weights.detach().numpy()
    text_weights = text_weights.detach().squeeze().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Visualizing image attention
    # Assuming image_input is a tensor of shape (3, H, W)
    image = image_input.permute(1, 2, 0).cpu().numpy()  # Convert to HxWx3 for matplotlib
    axs[0].imshow(image)
    axs[0].imshow(image_weights.reshape(image.shape[:2]), cmap='jet', alpha=0.5)  # Overlaying the heatmap
    axs[0].set_title("Image Attention")
    axs[0].axis('off')

    # Visualizing text attention
    tokens = [id2token.get(int(id), '') for id in text_input]
    y_pos = np.arange(len(tokens))
    axs[1].barh(y_pos, text_weights, align='center')
    axs[1].set_yticks(y_pos)
    axs[1].set_yticklabels(tokens)
    axs[1].invert_yaxis()  # labels read top-to-bottom
    axs[1].set_title("Text Attention")
    
    plt.show()

def visualize_cross_attention(image_path, text, model, tokenizer, transform_test, device):
    image = Image.open(image_path).convert('RGB')
    image = transform_test(image).unsqueeze(0).to(device)  # Assuming transform_test is defined

    encoding = tokenizer.encode_plus(
        text,
        return_tensors='pt',
        max_length=300,
        padding='max_length',
        truncation=True
    ).to(device)

    logits, attention_weights = model(image, encoding['input_ids'], encoding['attention_mask'])

    #logits, attention_weights = model(image, encoding['input_ids'], encoding['attention_mask'])
    print("Attention weights shape:", attention_weights.shape)
    print("Attention weights content:", attention_weights)


    # Check if attention weights are empty or malformed
    if attention_weights.nelement() == 0:
        print("Attention weights are empty. Check model output.")
        return logits

    # Ensure attention weights have at least two dimensions
    if attention_weights.ndim < 2:
        attention_weights = attention_weights.unsqueeze(0)  # Add a dimension if it's flat

    attention_weights = attention_weights.cpu().detach().numpy()

    # Display attention weights
    plt.figure(figsize=(10, 5))
    sns.heatmap(attention_weights, cmap='viridis', annot=True)
    plt.title('Attention Weights between Text and Image Features')
    plt.xlabel('Image Features')
    plt.ylabel('Text Features')
    plt.savefig("./heatmaps/heatmaps.png")  # Or save to a file

    return logits
