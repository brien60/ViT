import torch
from torch import nn
from torch import Tensor

import math

from dataclasses import dataclass

class MultiHeadAttention(nn.Module):
    """Models a mutli-head self-attention block for an Encoder"""

    def __init__(self, n_heads: int, embd_size: int) -> None:
        """Initializes the multi-head self-attention block.
        
        Args:
            n_heads: Number of attention heads in the attention head of each encoder.
            embd_size: Dimension size of the patch embeddings.
        """
        super().__init__()

        self.n_heads = n_heads
        self.embd_size = embd_size

        assert self.embd_size % self.n_heads == 0
        self.head_embd_size = self.embd_size // self.n_heads

        self.QKV = nn.Linear(self.embd_size, self.n_heads * 3*self.head_embd_size)

        # Linearly combines the output from each attention head
        self.W_O = nn.Linear(self.embd_size, self.embd_size)


    def forward(self, x: Tensor) -> Tensor:
        """Computes mutli-head self-attention over the input token sequence.
        
        Args:
            x: A tensor of shape (B, T, embd_size)

        Returns:
            A tensor of shape (B, T, embd_size) containing the contextualized representation of each input token.
        """
        B, T, embd_size = x.shape

        qkv = self.QKV(x).reshape(B, T, self.n_heads, -1).transpose(1, 2) # (B, n_heads, T, 3*head_embd_size)
        
        Q, K, V = qkv.split(self.head_embd_size, dim = -1) # (B, n_heads, T, head_embd_size)

        # (B, n_heads, T, T)
        attn_scores = torch.softmax((Q @ K.transpose(-1, -2)) / math.sqrt(self.head_embd_size), dim = -1)
        z = attn_scores @ V # (B, n_heads, T, head_embd_size)
        z = z.transpose(1, 2).reshape(B, T, -1) # (B, T, n_heads * head_embd_size) = (B, T, embd_size)
        z = self.W_O(z) # (B, T, embd_size)

        return z


class MLP(nn.Module):
    """Models a simple Multi-Layer Perceptron"""

    
    def __init__(
            self, 
            embd_size: int, 
            hidden_size: int, 
            output_size: int = None,
            dropout: float = 0.0
        ) -> None:
        """Initializes a MLP
        Args:
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons (if applicable)
        """
        
        super().__init__()
        self.layer1 = nn.Linear(embd_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if output_size is not None:
            self.layer2 = nn.Linear(hidden_size, output_size)
        else: 
            self.layer2 = nn.Linear(hidden_size, embd_size)

    
    def forward(self, x: Tensor) -> Tensor:
        """Maps input tokens through a two-layer MLP.
        
        Applies a linear transformation, a GELU activation, possibly a dropout, and a
        final linear transformation to each input token independently.
    
        Args:
            x: Tensor of shape (B, T, embd_size)

        Returns:
            Tensor of shape (B, T, output_size) if output_size was specified,
            otherwise (B, T, embd_size).
        """
        z = self.gelu(self.layer1(x))
        z = self.dropout(z)

        return self.layer2(z)


class Classifier(nn.Module):
    """Classifies image embeddings using an MLP or a fine-tuned linear layer."""

    def __init__(self, embd_size: int, hidden_size: int, output_size: int) -> None:
        """Initializes a Classifier

        Args:
            embd_size: Dimension of the image embedding
            output_size: Number of output neurons; the number of image classes
            hidden_size: Number of hidden neurons in the MLP
        """
        super().__init__()
        self.classifer = MLP(embd_size, hidden_size, output_size)
        self.classifier_fine_tune = nn.Linear(embd_size, output_size)

    
    def forward(self, x: Tensor, fine_tune: bool) -> Tensor:
        """Maps input image embedding to the logits for the image classes

        Args:
            x: Tensor of shape (B, embd_size)
            fine_tune: If true, classifies the image embedding with the fine-tuned layer,
            otherwise uses the MLP.
        
        Returns: 
            Tensor of shape (B, output_size)
        """
        
        if fine_tune: 
            return self.classifier_fine_tune(x)
        return self.classifer(x)
    

class EncoderBlock(nn.Module):
    """Models an Encoder block in the Vision Transformer."""

    def __init__(self, n_heads: int, embd_size: int, dropout: float = 0.0) -> None:
        """Initializes an Encoder block.
        
        Args:
            n_heads: Number of heads in the self-attention block
            embd_size: Dimension size of the patch embeddings
        """
        super().__init__()
        self.ln = nn.LayerNorm(embd_size)
        self.msa = MultiHeadAttention(n_heads, embd_size)
        self.mlp = MLP(embd_size, 4*embd_size, dropout=dropout)
        self.mlp_dropout = nn.Dropout(dropout)


    def forward(self, x: Tensor) -> Tensor:
        """"Applies multi-head self-attention and a MLP with pre-layer normalization,
        residual connections, and possibly dropout

        Layer-normalizes the input before applying multi-head self-attention, then adds the original input 
        to the attention output through a residual connection. 
        
        The resulting tensor is layer-normalized, and an MLP is applied independently to each token
        in the normalized output.
        
        Finally, a dropout is applied and the pre-normalized input to the MLP is added through a 
        second residual connection.

        Args:
            x: Input tensor of shape (B, T, embd_size), where T is the sequence length

        Returns:
            A tensor of shape (B, T, embd_size)
        """
        z_attn = self.msa(self.ln(x)) + x
        z = self.mlp(self.ln(z_attn))
        z = self.mlp_dropout(z)

        return z + z_attn

@dataclass
class ViTConfig:
    """Stores configuration parameters for a ViT.
    
    Atrributes:
        n_heads: Number of attention heads in the attention head of each encoder.
        embd_size: Dimension size of the patch embeddings.
        patch_size: Length of each image patch.
        num_encoders: Number of encoders in the ViT.
        dropout: Ratio of neurons that will be zeroed
        input_shape: Tuple containing the height, width, and channels of the input images.
        output_size: Number of image classes.
    """
    n_heads: int = 4
    embd_size: int = 128
    patch_size: int = 2
    dropout: float = 0.3
    num_encoders: int = 4
    input_shape: tuple = (32, 32, 3)
    output_size: int = 10


class ViT(nn.Module):
    """Models a Vision Transformer according to the original "An Image is Worth 16x16 Words" paper.""" 

    def __init__(self, config: ViTConfig) -> None:
        """Initializes a Vision Transformer.

        Args:
            config: The ViTConfig that will be used to initialize the Vision Transformer
        """
        super().__init__()
        self.config = config

        self.patch_size = config.patch_size
        H, W, C = config.input_shape

        assert H % config.patch_size == 0 and W % config.patch_size == 0, f"Invalid patch size {config.patch_size} for {H}x{W} image."
        num_patches = H*W // config.patch_size**2


        self.E = nn.Linear(config.patch_size**2 * C, config.embd_size, bias=False)
        self.E_pos = nn.Parameter(torch.randn(size = (1, num_patches+1, config.embd_size)))
        
        # the 1's are for the B and T dimensions
        self.CLS = nn.Parameter(torch.randn(size = (1, 1, config.embd_size)))

        self.encoders = nn.ModuleList(
            config.num_encoders * [EncoderBlock(config.n_heads, config.embd_size, config.dropout)]
        )

        self.classifier = Classifier(config.embd_size, 4*config.embd_size, config.output_size)


    def forward(self, x: Tensor, fine_tune: bool = False) -> Tensor:
        """Computes class logits from input images
        
        Args:
            x: Tensor of shape (B, C, H, W) containing the input images, where B is the batch size, C is the 
            number of channels, H is the height of the image, and W is the width of the image.
            fine_tune: Whether the function is being called during fine-tuning

        Returns:
            A tensor of shape (B, output_size) containing the logits for each input image, where output_size 
            is the number of image classes.
        """

        B, C, H, W = x.shape

        image_patches = torch.stack([
            x[:, :, i:i+self.patch_size, j:j+self.patch_size].reshape(B, -1) # (B, P*P*C)
            for j in range(0, W, self.patch_size)
            for i in range(0, H, self.patch_size)
        ], dim = 1) # (B, num_patches, P*P*C)

        patch_embeddings = self.E(image_patches) # (B, num_patches, embd_size)


        patch_embeddings = torch.cat(
            [self.CLS.expand(B, -1, -1), patch_embeddings],
            dim = 1
        ) # (B, num_patches+1, embd_size)
        z = patch_embeddings + self.E_pos
        
        for encoder in self.encoders:
            z = encoder(z)

        return self.classifier(z[:, 0], fine_tune)
