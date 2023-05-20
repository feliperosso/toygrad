""" 
Testing the feedforward and backward evaluation of the 
different operations allowed by the Tensor class defined
in engine.py
"""

# Load packages
import torch
import numpy as np
# Load toytorch engine
from toygrad.engine import Tensor

# - Aux function -
def to_torch(input):
    # Change numpy input to torch
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
    else:
        return torch.from_numpy(np.array([input])).squeeze()

# - Cross Entropy -
def test_cross_entropy():

    def single_cross_entropy(num_batches, nn_dim_out):
        # Initialize random inputs
        x_logits = np.random.uniform(-1, 1, size=(num_batches, nn_dim_out))
        y_logits = np.random.uniform(-1, 1, size=(num_batches, nn_dim_out))
        y_prob = np.exp(y_logits)/np.sum(np.exp(y_logits),
                                         axis=-1, keepdims=True)
        gradient = np.random.uniform(-1, 1, size=())

        # - My Tensor computation -
        xT_logits = Tensor(x_logits, requires_grad=True)
        yT_prob = Tensor(y_prob)
        gradientT = gradient
        outT = xT_logits.cross_entropy(yT_prob)
        outT.backward(gradientT)
        xT_logits_grad = xT_logits.grad

        # - Pytorch computation -
        xtorch_logits = torch.from_numpy(x_logits).requires_grad_(True)
        ytorch_prob = torch.from_numpy(y_prob)
        gradienttorch = torch.from_numpy(np.array([gradient])).squeeze()
        outtorch = torch.nn.functional.cross_entropy(
            xtorch_logits, ytorch_prob)
        outtorch.backward(gradienttorch)
        xtorch_logits_grad = xtorch_logits.grad

        # - Testing -
        # Forward
        assert torch.allclose(to_torch(outT.item), outtorch), \
            'Forward evaluation failed.'
        # Backward
        assert torch.allclose(to_torch(xT_logits_grad), xtorch_logits_grad), \
            'Backward evaluation failed.'

    # - Test cases -
    s1 = np.random.randint(1, 20, size=(6,))
    single_cross_entropy(s1[0], s1[1])
    single_cross_entropy(s1[2], s1[3])
    single_cross_entropy(s1[4], s1[5])

# - Tensor Addition -
def test_add():

    def test_single_add(size1, size2):
        """ Computes feedfoward and backward of addition of two tensors
        (allowing for broadcasting) and compares with Pytorch implementation.
        If succesful, nothing should happen. """
        # Initialize random sumands
        x1 = np.random.uniform(-1, 1, size=size1)
        x2 = np.random.uniform(-1, 1, size=size2)
        # Initialize random gradient
        gradient = np.random.uniform(-1, 1, size=(x1 + x2).shape)

        # - My Tensor computation -
        x1T = Tensor(x1, requires_grad=True)
        x2T = Tensor(x2, requires_grad=True)
        gradientT = gradient
        outT = x1T + x2T
        outT.backward(gradientT)
        x1T_grad, x2T_grad = x1T.grad, x2T.grad

        # - Pytorch computation -
        x1torch = torch.from_numpy(x1).requires_grad_(True)
        x2torch = torch.from_numpy(x2).requires_grad_(True)
        gradienttorch = torch.from_numpy(gradient)
        outtorch = x1torch + x2torch
        outtorch.backward(gradienttorch)
        x1torch_grad, x2torch_grad = x1torch.grad, x2torch.grad

        # - Testing -
        # Forward
        assert torch.allclose(to_torch(outT.item), outtorch), \
            'Forward evaluation failed.'
        # Backward
        assert torch.allclose(to_torch(x1T_grad), x1torch_grad), \
            'Backward evaluation failed.'
        assert to_torch(x1T_grad).shape == x1torch_grad.shape, \
            'Grad shapes do not match.'

        assert torch.allclose(to_torch(x2T_grad), x2torch_grad), \
            'Backward evaluation failed'
        assert to_torch(x2T_grad).shape == x2torch_grad.shape, \
            'Grad shapes do not match.'

    # Test cases (no broadcasting)
    test_single_add((7, ), (7, ))
    s1 = tuple(np.random.randint(1, 10, size=(3,)))
    test_single_add(s1, s1)
    s2 = tuple(np.random.randint(1, 10, size=(9,)))
    test_single_add(s2, s2)
    # Test cases (broadcasting)
    test_single_add((3, 1, 2), (1, 1))
    test_single_add((37, 13, 1, 7, 1), (14, 1, 9))
    test_single_add((1, 13, 5, 1), (2, 10, 1, 1, 33))

# - Matrix Multiplication (@) -
def test_matmul():

    def test_single_matmul(size1, size2):
        # Initialize random tensors
        v = np.random.uniform(-1, 1, size1)
        w = np.random.uniform(-1, 1, size2)
        # Initialize random gradient
        gradient = np.random.uniform(-1, 1, size=(v@w).shape)

        # - My Tensor computation -
        vT = Tensor(v, requires_grad=True)
        wT = Tensor(w, requires_grad=True)
        gradientT = gradient
        outT = vT.matmul(wT)
        outT.backward(gradientT)
        vT_grad, wT_grad = vT.grad, wT.grad

        # - Pytorch computation -
        vtorch = torch.from_numpy(v).requires_grad_(True)
        wtorch = torch.from_numpy(w).requires_grad_(True)
        gradienttorch = torch.from_numpy(gradient)
        outtorch = vtorch@wtorch
        outtorch.backward(gradienttorch)
        vtorch_grad, wtorch_grad = vtorch.grad, wtorch.grad

        # - Testing -
        # Forward
        assert torch.allclose(to_torch(outT.item), outtorch), \
            'Forward evaluation failed.'
        # Backward
        assert torch.allclose(to_torch(vT_grad), vtorch_grad), \
            'Backward evaluation failed.'
        assert to_torch(vT_grad).shape == vtorch_grad.shape, \
            'Grad shapes do not match.'

        assert torch.allclose(to_torch(wT_grad), wtorch_grad), \
            'Backward evaluation failed'
        assert to_torch(wT_grad).shape == wtorch_grad.shape, \
            'Grad shapes do not match.'

    # - Test cases -
    # Vector-Vector dot product
    s1 = tuple(np.random.randint(1, 20, size=(1,)))
    test_single_matmul(s1, s1)
    s2 = tuple(np.random.randint(1, 20, size=(1,)))
    test_single_matmul(s2, s2)
    # Vector-Matrix broadcasted product
    s3 = np.random.randint(1, 20, size=(7,))
    test_single_matmul((s3[0], ), (s3[0], s3[1]))  # No broadcasting
    test_single_matmul((s3[2], ), (s3[6], s3[5], s3[4],
                       s3[2], s3[3]))  # Broadcasting
    # Matrix-Vector product
    s4 = np.random.randint(1, 20, size=(7,))
    test_single_matmul((s4[0], s4[1]), (s4[1], ))  # No broadcasting
    test_single_matmul((s4[6], s4[4], s4[5], s4[2], s4[3]),
                       (s4[3], ))  # Broadcasting
    # Matrix-Matrix product
    s5 = np.random.randint(1, 20, size=(6, ))
    test_single_matmul((s5[0], s5[1]), (s5[1], s5[2]))  # No broadcasting
    test_single_matmul((s5[3], s5[4]), (s5[4], s5[5]))  # No broadcasting
    test_single_matmul((7, 1, 9) + (3, 2), (1, 5, 9) + (2, 4))  # Broadcasting
    test_single_matmul((2, 4, 1) + (4, 8), (1, 5) + (8, 6))  # Broadcasting
    test_single_matmul((3, 11) + (1, 3), (7, 3, 1) + (3, 9))  # Broadcasting
    test_single_matmul((5, 1, 3, 2, 4), (3, 1, 4, 3))
    test_single_matmul((358, 3275), (3275, 345))

# - Elementwise mul (*) -
def test_mul():

    def test_single_mul(size1, size2):
        # Initialize random tensors
        v = np.random.uniform(-1, 1, size1)
        w = np.random.uniform(-1, 1, size2)
        # Initialize random gradient
        gradient = np.random.uniform(-1, 1, size=(v*w).shape)

        # - My Tensor computation -
        vT = Tensor(v, requires_grad=True)
        wT = Tensor(w, requires_grad=True)
        gradientT = gradient
        outT = vT*wT
        outT.backward(gradientT)
        vT_grad, wT_grad = vT.grad, wT.grad

        # - Pytorch computation -
        vtorch = torch.from_numpy(v).requires_grad_(True)
        wtorch = torch.from_numpy(w).requires_grad_(True)
        gradienttorch = torch.from_numpy(gradient)
        outtorch = vtorch*wtorch
        outtorch.backward(gradienttorch)
        vtorch_grad, wtorch_grad = vtorch.grad, wtorch.grad

        # - Testing -
        # Forward
        assert torch.allclose(to_torch(outT.item), outtorch), \
            'Forward evaluation failed.'
        # Backward
        assert torch.allclose(to_torch(vT_grad), vtorch_grad), \
            'Backward evaluation failed.'
        assert to_torch(vT_grad).shape == vtorch_grad.shape, \
            'Grad shapes do not match.'

        assert torch.allclose(to_torch(wT_grad), wtorch_grad), \
            'Backward evaluation failed'
        assert to_torch(wT_grad).shape == wtorch_grad.shape, \
            'Grad shapes do not match.'

    # - Test cases -
    # No broadcasting
    s1 = np.random.randint(1, 20, size=(6,))
    test_single_mul((s1[0], s1[1], s1[2]), (s1[0], s1[1], s1[2]))
    test_single_mul((s1[3], s1[4], s1[5]), (s1[3], s1[4], s1[5]))
    # Scalar-Tensor multiplication
    s2 = np.random.randint(1, 20, size=(5,))
    test_single_mul((), (s2[0], s2[1]))
    test_single_mul((s2[2], s2[3], s2[4]), ())
    # Lots of broadcasting
    test_single_mul((6, 3, 1, 8, 1), (3, 7, 1, 9))
    test_single_mul((1, 5, 1, 1), (9, 4, 4, 1, 2, 1))

# - ReLU -
def test_relu():

    def test_single_relu(size):
        # Initialize random tensor and gradient
        x = np.random.uniform(-1, 1, size)
        gradient = np.random.uniform(-1, 1, size)

        # - My Tensor computation -
        xT = Tensor(x, requires_grad=True)
        gradientT = gradient
        outT = xT.relu()
        outT.backward(gradientT)
        xT_grad = xT.grad

        # - Pytorch computation -
        xtorch = torch.from_numpy(x).requires_grad_(True)
        gradienttorch = torch.from_numpy(gradient)
        outtorch = torch.nn.functional.relu(xtorch)
        outtorch.backward(gradienttorch)
        xtorch_grad = xtorch.grad

        # - Testing -
        # Forward
        assert torch.allclose(to_torch(outT.item), outtorch), \
            'Forward evaluation failed.'
        # Backward
        assert torch.allclose(to_torch(xT_grad), xtorch_grad), \
            'Backward evaluation failed.'

    # - Test cases -
    s1 = tuple(np.random.randint(1, 10, size=(3,)))
    test_single_relu(s1)
    s2 = tuple(np.random.randint(1, 10, size=(3,)))
    test_single_relu(s2)

# - Reshape -
def test_reshape():

    def test_single_reshape(in_size, fi_size):
        x = np.random.uniform(-1, 1, in_size)
        gradient = np.random.uniform(-1, -1, fi_size)

        # - My Tensor computation -
        xT = Tensor(x, requires_grad=True)
        gradientT = gradient
        outT = xT.reshape(fi_size)
        outT.backward(gradientT)
        xT_grad = xT.grad

        # - Pytorch computation -
        xtorch = torch.from_numpy(x).requires_grad_(True)
        gradienttorch = torch.from_numpy(gradient)
        outtorch = xtorch.reshape(fi_size)
        outtorch.backward(gradienttorch)
        xtorch_grad = xtorch.grad

        # - Testing -
        # Forward
        assert torch.allclose(to_torch(outT.item), outtorch), \
                'Forward evaluation failed.'
        assert to_torch(outT.item).shape == outtorch.shape, \
                'Forward shapes do not match.'
        # Backward
        assert torch.allclose(to_torch(xT_grad), xtorch_grad), \
                'Backward evaluation failed.'
        assert to_torch(xT_grad).shape == xtorch_grad.shape, \
                'Backward shapes do not match.'
    
    # - Test cases -
    s1 = np.random.randint(1, 10, size=(3, ))
    test_single_reshape((s1[0], s1[1], s1[2]), (s1[0]*s1[1]*s1[2]))
    s2 = np.random.randint(1, 10, size=(4, ))
    test_single_reshape((s2[0]*s2[1], s2[2]*s2[3]), (s2[0], s2[1], s2[2], s2[3]))
    s3 = np.random.randint(1, 10, size=(4, ))
    test_single_reshape((s3[0]*s3[1], s3[2], s3[3]), (s3[2], s3[0], s3[1], s3[3]))

# - Sum -
def test_sum():

    def test_single_sum(input_size, dims_sumed=None):
        if dims_sumed == None:
            dims_sumed = tuple(range(len(input_size)))
        x = np.random.uniform(-1, 1, input_size)
        out = x.sum(dims_sumed)
        gradient = np.random.uniform(-1, 1, out.shape)

        # - My Tensor computation -
        xT = Tensor(x, requires_grad=True)
        gradientT = gradient 
        outT = xT.sum(dims_sumed)
        outT.backward(gradientT)
        xT_grad = xT.grad

        # - Pytorch computation -
        xtorch = torch.from_numpy(x).requires_grad_(True)
        gradienttorch = torch.from_numpy(gradient)
        outtorch = xtorch.sum(dims_sumed)
        outtorch.backward(gradienttorch)
        xtorch_grad = xtorch.grad

        # - Testing -
        # Forward
        assert torch.allclose(to_torch(outT.item), outtorch), \
            "Forward evaluation failed."
        assert to_torch(outT.item).shape == outtorch.shape, \
            "Forward shapes do not match."
        # Backward
        assert torch.allclose(to_torch(xT_grad), xtorch_grad), \
                'Backward evaluation failed.'
        assert to_torch(xT_grad).shape == xtorch_grad.shape, \
                'Backward shapes do not match.'
    
    # - Test cases -
    s1 = tuple(np.random.randint(1, 10, size=(3, )))
    test_single_sum(s1, (0, 1))
    s2 = tuple(np.random.randint(1, 10, size=(6, )))
    test_single_sum(s2, (5, 1, 3))
    s3 = tuple(np.random.randint(1, 10, size=(4, )))
    test_single_sum(s3)

# - Test a combination of the above operations -
def test_full():
    # Initialize random inputs
    x1 = np.random.uniform(-1, 1, (9, 3, 1, 2) + (4, 5))
    x2 = np.random.uniform(-1, 1, (3, 7, 1) + (5, 2))
    x3 = np.random.uniform(-1, 1, (9, 1, 7, 2, 1, 2))
    x4 = np.random.uniform(-1, 1, (1, 3, 7, 2, 1, 1))
    x5 = np.random.uniform(-1, 1, ())

    # - My Tensor computation -
    x1T = Tensor(x1, requires_grad=True)
    x2T = Tensor(x2, requires_grad=True)
    x3T = Tensor(x3, requires_grad=True)
    x4T = Tensor(x4, requires_grad=True)
    x5T = Tensor(x5, requires_grad=True)

    outT = ((x1T.matmul(x2T) + x3T)*x4T).relu()*x5T
    gradientT = np.random.uniform(-1, 1, size=outT.item.shape)
    outT.backward(gradientT)

    # - Pytorch computation -
    x1torch = to_torch(x1).requires_grad_(True)
    x2torch = to_torch(x2).requires_grad_(True)
    x3torch = to_torch(x3).requires_grad_(True)
    x4torch = to_torch(x4).requires_grad_(True)
    x5torch = to_torch(x5).requires_grad_(True)

    outtorch = ((x1torch.matmul(x2torch) + x3torch)*x4torch).relu()*x5torch
    gradienttorch = to_torch(gradientT)
    outtorch.backward(gradienttorch)

    # - Testing -
    assert torch.allclose(to_torch(x1T.grad), x1torch.grad)
    assert torch.allclose(to_torch(x2T.grad), x2torch.grad)
    assert torch.allclose(to_torch(x3T.grad), x3torch.grad)
    assert torch.allclose(to_torch(x4T.grad), x4torch.grad)
    assert torch.allclose(to_torch(x5T.grad), x5torch.grad)
