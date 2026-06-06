from abc import ABC, abstractmethod
from math import prod
import cupy as np
from numpy.lib.stride_tricks import sliding_window_view


def im2col(entree_pad, kH, kW):
    """
    Transforme un batch d'images padees en matrice de colonnes.

    entree_pad : (N, C, H_pad, W_pad)
    Retour     : (N, C*kH*kW, out_H*out_W)   ou out_H = H_pad - kH + 1
    """
    N, C, H_pad, W_pad = entree_pad.shape
    out_H = H_pad - kH + 1
    out_W = W_pad - kW + 1
    windows = sliding_window_view(entree_pad, (kH, kW), axis=(2, 3))
    windows = windows.transpose(0, 1, 4, 5, 2, 3).reshape(N, C * kH * kW, out_H * out_W)
    return windows


def col2im(cols, N, C, H_pad, W_pad, kH, kW):
    """
    Operation inverse pour le backward : reconstruit le tenseur depad
    en sommant les contributions de chaque fenetre.

    cols    : (N, C*kH*kW, out_H*out_W)
    Retour  : (N, C, H_pad, W_pad)
    """
    out_H = H_pad - kH + 1
    out_W = W_pad - kW + 1
    cols_reshaped = cols.reshape(N, C, kH, kW, out_H, out_W)
    sortie = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    for i in range(kH):
        for j in range(kW):
            sortie[:, :, i:i + out_H, j:j + out_W] += cols_reshaped[:, :, i, j, :, :]
    return sortie



class Couche(ABC):
    @abstractmethod
    def forward(self, entree):
        pass

    @abstractmethod
    def backward(self, gradient, learning_rate):
        pass


class Couche_convolution(Couche):
    """
    Convolution vectorisee via im2col + un seul produit matriciel.
    Entrees attendues : (N, C, H, W)
    """

    def __init__(self, nb_filtre, dim_filtre, activation, dim_entree):
        # dim_filtre = (C_in, kH, kW)
        assert dim_filtre[1] % 2 == 1 and dim_filtre[2] % 2 == 1, \
            "Les dimensions du filtre doivent etre impaires (padding 'same')"
        self.activation = activation
        self.dim_filtre = dim_filtre
        self.nb_filtre = nb_filtre
        self.C_in, self.kH, self.kW = dim_filtre


        self.poids = activation.init_poids((nb_filtre,) + tuple(dim_filtre))
        self.biais = np.zeros(nb_filtre, dtype=np.float32)

        self.pad_h = (self.kH - 1) // 2
        self.pad_w = (self.kW - 1) // 2

        self.dim_sortie = (nb_filtre, dim_entree[1], dim_entree[2])

        self.cols = None
        self.score = None
        self.entree_shape = None

    def forward(self, entree):
        # entree : (N, C, H, W)
        N, C, H, W = entree.shape
        assert C == self.C_in, f"Canaux attendus : {self.C_in}, recus : {C}"

        entree_pad = np.pad(entree,
                            ((0, 0), (0, 0),
                             (self.pad_h, self.pad_h),
                             (self.pad_w, self.pad_w)),
                            mode="constant")

        self.cols = im2col(entree_pad, self.kH, self.kW)

        W_flat = self.poids.reshape(self.nb_filtre, -1)

        out = np.einsum('fk,nkp->nfp', W_flat, self.cols)
        out = out + self.biais.reshape(1, -1, 1)
        out = out.reshape(N, self.nb_filtre, H, W)

        self.score = out
        self.entree_shape = (N, C, H + 2 * self.pad_h, W + 2 * self.pad_w)
        return self.activation.calcul(out)

    def backward(self, gradient, learning_rate):
        # gradient : (N, nb_filtre, H, W)
        N, _, H, W = gradient.shape

        dZ = gradient * self.activation.derive(self.score)
        dZ_flat = dZ.reshape(N, self.nb_filtre, -1)
        dW_flat = np.einsum('nfp,nkp->fk', dZ_flat, self.cols) / N
        dW = dW_flat.reshape(self.poids.shape)
        db = dZ.sum(axis=(0, 2, 3)) / N
        W_flat = self.poids.reshape(self.nb_filtre, -1)
        dcols = np.einsum('fk,nfp->nkp', W_flat, dZ_flat)
        N_, C_in, H_pad, W_pad = self.entree_shape
        dX_pad = col2im(dcols, N_, C_in, H_pad, W_pad, self.kH, self.kW)
        if self.pad_h > 0 or self.pad_w > 0:
            dX = dX_pad[:, :, self.pad_h:H_pad - self.pad_h,
                              self.pad_w:W_pad - self.pad_w]
        else:
            dX = dX_pad
        self.poids -= learning_rate * dW.astype(self.poids.dtype)
        self.biais -= learning_rate * db.astype(self.biais.dtype)

        return dX


class Couche_max_pooling(Couche):
    """
    Max-pooling vectorise via sliding_window_view.
    """

    def __init__(self, dim_filtre, dim_entree):
        self.dim_filtre = dim_filtre  # (pH, pW)
        self.pH, self.pW = dim_filtre
        self.dim_sortie = (dim_entree[0], dim_entree[1] // self.pH,
                                          dim_entree[2] // self.pW)
        self.masque = None
        self.entree_shape = None

    def forward(self, entree):
        # entree : (N, C, H, W)
        N, C, H, W = entree.shape
        assert H % self.pH == 0 and W % self.pW == 0, "H et W doivent etre divisibles par la taille du filtre"
        self.entree_shape = entree.shape
        out_H, out_W = H // self.pH, W // self.pW

        reshaped = entree.reshape(N, C, out_H, self.pH, out_W, self.pW)
        reshaped = reshaped.transpose(0, 1, 2, 4, 3, 5)
        flat = reshaped.reshape(N, C, out_H, out_W, self.pH * self.pW)
        idx_max = np.argmax(flat, axis=-1)
        sortie = np.take_along_axis(flat, idx_max[..., None], axis=-1).squeeze(-1)
        masque_flat = np.zeros_like(flat)
        np.put_along_axis(masque_flat, idx_max[..., None], 1, axis=-1)
        # Retour au format image
        masque = masque_flat.reshape(N, C, out_H, out_W, self.pH, self.pW)
        masque = masque.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)
        self.masque = masque
        return sortie

    def backward(self, gradient, learning_rate):
        # gradient : (N, C, out_H, out_W)
        grad_etire = np.repeat(gradient, self.pH, axis=2)
        grad_etire = np.repeat(grad_etire, self.pW, axis=3)
        return grad_etire * self.masque


class Couche_Aplatissement(Couche):
    def __init__(self, dim_entree):
        self.dim_entree = dim_entree  # (C, H, W)
        self.dim_sortie = (prod(dim_entree),)
        self.batch_shape = None

    def forward(self, entree):
        # entree : (N, C, H, W) -> (N, C*H*W)
        self.batch_shape = entree.shape
        return entree.reshape(entree.shape[0], -1)

    def backward(self, gradient, learning_rate):
        return gradient.reshape(self.batch_shape)


class CoucheDense(Couche):
    def __init__(self, activation, n_entree, n_sortie):
        self.activation = activation
        if isinstance(n_entree, tuple):
            n_entree = n_entree[0]
        self.n_entree = n_entree
        self.dim_sortie = (n_sortie,)
        self.poids = activation.init_poids((n_entree, n_sortie))
        self.biais = np.zeros(n_sortie, dtype=np.float32)
        self.entree = None
        self.score = None

    def forward(self, entree):
        # entree : (N, n_entree)
        self.entree = entree
        self.score = entree @ self.poids + self.biais
        return self.activation.calcul(self.score)

    def backward(self, gradient, learning_rate):
        # gradient : (N, n_sortie)
        N = gradient.shape[0]
        dZ = gradient * self.activation.derive(self.score)

        dW = self.entree.T @ dZ / N
        db = dZ.sum(axis=0) / N
        gradient_precedent = dZ @ self.poids.T

        self.poids -= learning_rate * dW.astype(self.poids.dtype)
        self.biais -= learning_rate * db.astype(self.biais.dtype)
        return gradient_precedent
