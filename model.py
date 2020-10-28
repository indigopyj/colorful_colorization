import torch; torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F

from modules.cielab import ABGamut, CIELAB, DEFAULT_CIELAB
from  modules.annealed_mean_decode_q import AnnealedMeanDecodeQ
from  modules.get_class_weights import GetClassWeights
from  modules.rebalance_loss import RebalanceLoss
from  modules.soft_encode_ab import SoftEncodeAB
from  modules.vgg_segmentation_network import VGGSegmentationNetwork


class ColorizationNetwork(nn.Module):
    """Wrapper class implementing input encoding, output decoding and class
       rebalancing.

    This class is independent of the concrete underlying network (by default
    the VGG style architecture described by Zhang et al.) so that the latter
    can in principle be exchanged for another network by modifying the
    `base_network` attribute.

    """

    def __init__(self,
                 annealed_mean_T=0.38,
                 class_rebal_lambda=0.5,
                 device='cuda'):
        """
        Construct the network.

        Args:
            annealed_mean_T (float):
                Annealed mean temperature parameter, should be between 0.0 and
                1.0. Lower values result in less saturated but more spatially
                consistent outputs.
            class_rebal_lambda (float, optional):
                Class rebalancing parameter, class rebalancing is NOT enabled
                by default (i.e. when this is `None`). Zhang et al. recommend
                setting this parameter to 0.5.
            device (str):
                Device on which to run the network (i.e. `'cpu'` or `'cuda'`),
                note that this can not be changed post construction.

        """

        super().__init__()



        self.base_network = VGGSegmentationNetwork(ABGamut.EXPECTED_SIZE) # Expected_size = 313


        self.device = device

        # en-/decoding
        self.encode_ab = SoftEncodeAB(DEFAULT_CIELAB,
                                      device=self.device)

        self.decode_q = AnnealedMeanDecodeQ(DEFAULT_CIELAB,
                                            T=annealed_mean_T,
                                            device=self.device)



        # rebalancing
        self.class_rebal_lambda = class_rebal_lambda

        if class_rebal_lambda is not None:
            self.get_class_weights = GetClassWeights(DEFAULT_CIELAB,
                                                     lambda_=class_rebal_lambda,
                                                     device=self.device)

            self.rebalance_loss = RebalanceLoss.apply

        # move to device
        self.to(self.device)

    def forward(self, sketch_img, color_img):
        """"Network forward pass.

        img (torch.Tensor):
            A tensor of shape `(n, 1, h, w)` where `n` is the size of the
            batch to be predicted and `h` and `w` are image dimensions.
            Must be located on the same device as this network. The
            images should be Lab lightness channels.

        Returns:
            If this network is in training mode: A tuple containing two tensors
            of shape `(n, Q, h, w)`, where `Q` is the number of ab output bins.
            The first element of this tuple is the predicted ab bin distribution
            and the second the soft encoded ground truth ab bin distribution.

            Else, if this model is in evaluation mode: A tensor of shape
            `(n, 1, h, w)` containing the predicted ab channels.

        """

        # label transformation
        if self.training:
            return self._forward_encode(sketch_img, color_img)
        else:
            return self._forward_decode(sketch_img)

    def _forward_encode(self, sketch_img, color_img):
        l, ab = color_img[:, :1, :, :], color_img[:, 1:, :, :]

        l_norm = self._normalize_l(l)

        q_pred = self.base_network(l_norm)

        # downsample and encode labels
        ab = F.interpolate(ab, size=q_pred.shape[2:])
        q_actual = self.encode_ab(ab)
        # rebalancing
        if self.class_rebal_lambda is not None:
            color_weights = self.get_class_weights(q_actual)
            q_pred = self.rebalance_loss(q_pred, color_weights)

        return q_pred, q_actual

    def _forward_decode(self, img):
        l = img

        l_norm = self._normalize_l(l)

        q_pred = self.base_network(l_norm)

        ab_pred = self.decode_q(q_pred)



        return ab_pred

    def _normalize_l(self, l):
        l_norm = l - CIELAB.L_MEAN
        return l_norm


