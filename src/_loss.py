from torch import mean, Tensor, mm


def compute_content_loss(generated_img_features, content_img_features) -> Tensor :
    return mean((generated_img_features-content_img_features)**2) #MSE

def compute_style_loss(generated_img_features, style_img_features) -> Tensor:

    _,channel,height,width=generated_img_features.shape

    G=mm(generated_img_features.view(channel,height*width),generated_img_features.view(channel,height*width).t())
    A=mm( style_img_features.view(channel,height*width), style_img_features.view(channel,height*width).t())

    return mean((G-A)**2)


class Loss:
    def __init__(self,alpha : float = .5, beta : float = .5) :
        self.alpha = alpha
        self.beta = beta

    def __call__(self, generated_img_features, style_img_features, content_img_features) -> Tensor:
        style_loss=content_loss=0
        for gen,cont,style in zip(generated_img_features,content_img_features,style_img_features):

            content_loss+=compute_content_loss(gen,cont)
            style_loss+=compute_style_loss(gen,style)

        return self.alpha*content_loss + self.beta*style_loss