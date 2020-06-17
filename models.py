import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable


class CategoryClassifier(nn.Module):
    """
    CategoryClassifier.
    """

    def __init__(self, args, vocab_size, num_class):
        super(CategoryClassifier, self).__init__()
        self.args = args

        self.enc_image_size = args.encoded_image_size
        self.embed_dim = args.embed_dim
        self.num_class = num_class

        # image
        self.mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)  # pretrained ImageNet MobileNet-v2
        # Remove last layer
        modules = list(self.mobilenet_v2.children())[0]
        self.mobilenet_v2 = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.max_pool = nn.MaxPool2d(args.encoded_image_size, padding=1)
        c, h, w = self.mobilenet_v2[-1][0].out_channels, args.encoded_image_size, args.encoded_image_size
        self.fc1 = nn.Linear(in_features=c * w * h, out_features=self.embed_dim, bias=True)

        # text
        # window_sizes means that how many words a pattern covers
        self.window_sizes = args.window_sizes
        # n_filters means that how many patterns to cover
        self.n_filters = args.n_filters
        self.embedding = nn.Embedding(vocab_size, args.embed_dim)  # embedding layer
        # Use nn.ModuleList to register each sub-modules.
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, self.n_filters, (w, self.embed_dim)) for w in self.window_sizes])
        '''
            self.conv13 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim))
            self.conv14 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim))
            self.conv15 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim))
        '''
        self.dropout = nn.Dropout(args.dropout_p)
        self.fc2 = nn.Linear(len(self.window_sizes) * self.n_filters, self.embed_dim)
        self.fc3 = nn.Linear(self.embed_dim + self.embed_dim, self.num_class)

        self.init_weights()
        self.fine_tune()
        self.fine_tune_embeddings()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (batch_size, in_channels, embedding_dim)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, images, texts):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :param texts: texts, a tensor of dimensions (batch_size, max_len)
        :param device: device
        :return: prediction values
        """

        # image features
        x1 = self.mobilenet_v2(images)  # (batch_size, 1280, image_size/32, image_size/32)
        x1 = self.max_pool(x1)  # (batch_size, 1280, encoded_image_size, encoded_image_size)
        bs = x1.size(0)
        x1 = x1.view(bs, -1)
        x1 = self.dropout(x1)
        x1 = self.fc1(x1)  # (batch_size, encoded_image_size, encoded_image_size, 512)

        # text features
        x2 = self.embedding(texts)  # (batch_size, vocab_size, embed_dim)
        if self.args.static:
            x2 = Variable(x2)
        x2 = x2.unsqueeze(1)  # (batch_size, in_channels, vocab_size, embed_dim)
        x2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channels, vocab_size), ...]*len(window_sizes)
        x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x2]  # [(batch_size, out_channels), ...]*len(window_sizes)
        x2 = torch.cat(x2, 1)
        '''
            x1 = self.conv_and_pool(x,self.conv13) #(batch_size, out_channels)
            x2 = self.conv_and_pool(x,self.conv14) #(batch_size, out_channels)
            x3 = self.conv_and_pool(x,self.conv15) #(batch_size, out_channels)
            x = torch.cat((x1, x2, x3), 1) # (batch_size, len(window_sizes)*out_channels)
        '''
        x2 = self.dropout(x2)  # (batch_size, len(window_sizes)*out_channels)
        x2 = self.fc2(x2)  # (batch_size, 512)

        # concatenate two features
        out = torch.cat((x1, x2), dim=1)
        out = self.fc3(out)

        return out

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.mobilenet_v2.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.mobilenet_v2.children())[-3:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
