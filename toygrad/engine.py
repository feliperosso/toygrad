"""
engine.py
"""

# Load packages
from __future__ import annotations
import numpy as np

class Tensor:

    def __init__(self, value: np.ndarray, children: tuple=(), requires_grad: bool=False) -> None:
        # Forward
        self.item = value if not isinstance(value, int) else np.array([value])
        # Backward
        self.requires_grad = requires_grad
        if requires_grad:
            self.children = set(children)
            self.grad = 0
            self._backward = lambda: None
    
    # - Aux functions -
    def check_instance(self, other) -> None:
        """ Checks if 'other' element is an instance of Tensor class """
        if not isinstance(other, Tensor):
            raise Exception("The 'other' element is not an instance of the Tensor class.")
    
    @staticmethod
    def crop_extra_dims(pre_grad: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """" Given a pre_grad, it removes additional length 1 dims
        on the left that can sometimes be present due to broadcasting"""
        # Remove additional length 1 dims, if present
        dim_diff = len(pre_grad.shape) - len(reference.item.shape)
        if dim_diff > 0:
            return pre_grad.squeeze(tuple(range(dim_diff)))
        else:
            return pre_grad

    # - Cross Entropy -
    def cross_entropy(self, y_prob: Tensor) -> Tensor:
        """ self is nn_logits: (num_batches, nn_dim_out)
            y_prob: (num_batches, nn_dim_out) """
        self.check_instance(y_prob)
        # Forward
        batch_loss = np.log(np.sum(np.exp(self.item), axis=-1)) - np.einsum('ij,ij->i', y_prob.item, self.item)
        out = Tensor(batch_loss.mean())
        if self.requires_grad:
            # Update parameters
            out.__init__(out.item, children=(self, y_prob), requires_grad=True)
            # Backward
            def _backward():
                num_batches = self.item.shape[0]
                self.grad += ( (np.exp(self.item).T/np.sum(np.exp(self.item), axis=-1)).T - y_prob.item)*out.grad/num_batches
            out._backward = _backward            
        return out

    # - Addition and Subtraction (+/-) -
    def __add__(self, other: Tensor) -> Tensor:
        self.check_instance(other)
        # Forward
        out = Tensor(self.item + other.item)
        if self.requires_grad or other.requires_grad:
            # Update parameters
            out.__init__(out.item, children=(self, other), requires_grad=True)
            # Backward
            def _backward():
                # - Auxiliary function that computes gradient of each summand -
                def sum_grad(sumand: np.ndarray, sum_result: np.ndarray, gradient: np.ndarray) -> np.ndarray:
                    pre_grad = np.ones(sumand.shape)*gradient
                    # Account for having a sumand with smaller shape than sum_result 
                    # which can happen when broadcasting
                    sumand_shape = sumand.shape
                    sum_result_shape = sum_result.shape
                    sumand_shape = tuple([-1]*(len(sum_result_shape) - len(sumand_shape))) + sumand_shape
                    # Store dimensions affected by broadcasting
                    broad_dims = () 
                    for dim, bool in enumerate(np.equal(sumand_shape, sum_result_shape)):
                        if not bool:
                            broad_dims += (dim, )
                    # If there was broadcasting, sum accordingly
                    if len(broad_dims) > 0:
                        out = np.sum(pre_grad, axis=broad_dims).reshape(sumand.shape)
                    else:
                        out = pre_grad
                    return out
                # -
                # Update grads (if required)
                if self.requires_grad:
                    self.grad += sum_grad(self.item, out.item, out.grad)
                if other.requires_grad:
                    other.grad += sum_grad(other.item, out.item, out.grad)
            # Update _backward function of out                    
            out._backward = _backward            
        return out
    
    def __radd__(self, other: Tensor) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor) -> Tensor:
        return self + (-other)

    def __rsub__(self, other: Tensor) -> Tensor:
        return other + (-self)

    # - Get Item (slicing) -
    def __getitem__(self, slicing) -> Tensor:
        # Forward
        out = Tensor(self.item[slicing])
        # Backward
        if self.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, ), requires_grad=True)
            # Backward
            def _backward():
                grad = np.zeros(self.item.shape)
                grad[slicing] = out.grad
                self.grad += grad
            out._backward = _backward
        return out

    # - Log -
    def log(self) -> Tensor:
        # Forward
        out = Tensor(np.log(self.item))
        if self.requires_grad:
            # Update out tensor parameters
            out.__init__(out.item, children=(self, ), requires_grad=True)
            # Backward
            def _backward():
                self.grad += out.grad/self.item
            out._backward = _backward
        return out

    # - MatMul (@) -
    def matmul(self, other: Tensor) -> Tensor:
        self.check_instance(other)
        # Forward
        out = Tensor(self.item@other.item)
        if self.requires_grad or other.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, other), requires_grad=True)
            # Backward
            def _backward():
                # Vector-Vector dot product
                if len(self.item.shape) == len(other.item.shape) == 1:
                    if self.requires_grad:
                        self.grad += other.item*out.grad
                    if other.requires_grad:
                        other.grad += self.item*out.grad
                # Vector-Matrix broadcasted product
                if len(self.item.shape) == 1 and len(other.item.shape) > 1:
                    if self.requires_grad:
                        pre = out.grad[..., None, :]@np.swapaxes(other.item, -1, -2)
                        pre_grad = np.sum(pre.reshape(-1, self.item.shape[0]), axis=0)
                        self.grad += self.crop_extra_dims(pre_grad, self)
                    if other.requires_grad:
                        pre_grad = self.item[..., None]*out.grad[..., None, :]
                        other.grad += self.crop_extra_dims(pre_grad, other)
                # Matrix-Vector broadcasted product
                if len(self.item.shape) > 1 and  len(other.item.shape) == 1:
                    if self.requires_grad:
                        pre_grad = out.grad[..., None]*other.item[None]
                        self.grad += self.crop_extra_dims(pre_grad, self)
                    if other.requires_grad:
                        pre = (np.swapaxes(self.item, -1, -2)@out.grad[..., None]).squeeze()
                        pre_grad = np.sum(pre.reshape(-1, other.item.shape[0]), axis=0)
                        other.grad += self.crop_extra_dims(pre_grad, other)
                # Matrix-Matrix product
                if len(self.item.shape) > 1 and len(other.item.shape) > 1:
                    # Identify broadcasted dims we need to sum over
                    self_b_shape = self.item.shape[:-2]
                    other_b_shape = other.item.shape[:-2]
                    out_grad_b_shape = out.grad.shape[:-2]
                    # Enlarge shapes if required
                    self_b_shape = tuple([-1]*(len(out_grad_b_shape) - len(self_b_shape))) + self_b_shape
                    other_b_shape = tuple([-1]*(len(out_grad_b_shape) - len(other_b_shape))) + other_b_shape
                    # Find enlarged dims
                    self_enlarged_dims, other_enlarged_dims = (), ()
                    for dim, (se_d, ot_d, ou_d) in enumerate(zip(self_b_shape, other_b_shape, out_grad_b_shape)):
                        if se_d != ou_d:
                            self_enlarged_dims += (dim, )
                        if ot_d != ou_d:
                            other_enlarged_dims += (dim, )
                    if self.requires_grad:
                        pre_grad = np.sum(out.grad@np.swapaxes(other.item, -1, -2), self_enlarged_dims, keepdims=True)
                        self.grad += self.crop_extra_dims(pre_grad, self)
                    if other.requires_grad:
                        pre_grad = np.sum(np.swapaxes(self.item, -1, -2)@out.grad, other_enlarged_dims, keepdims=True)
                        other.grad += self.crop_extra_dims(pre_grad, other)
            # Update _backward function of out                    
            out._backward = _backward
        return out

    # - Elementwise mul (*) -
    def __mul__(self, other: Tensor) -> Tensor:
        self.check_instance(other)
        # Forward
        out = Tensor(self.item*other.item)
        if self.requires_grad or other.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, other), requires_grad=True)
            # Backward
            def _backward():
                # Identify broadcasted dims we need to sum over
                self_shape = self.item.shape
                other_shape = other.item.shape
                out_grad_shape = out.grad.shape
                # Enlarge shapes if required
                self_shape = tuple([-1]*(len(out_grad_shape) - len(self_shape))) + self_shape
                other_shape = tuple([-1]*(len(out_grad_shape) - len(other_shape))) + other_shape
                # Find enlarged dims
                self_enlarged_dims, other_enlarged_dims = (), ()
                for dim, (se_d, ot_d, ou_d) in enumerate(zip(self_shape, other_shape, out_grad_shape)):
                    if se_d != ou_d:
                        self_enlarged_dims += (dim, )
                    if ot_d != ou_d:
                        other_enlarged_dims += (dim, )
                if self.requires_grad:
                    pre_grad = np.sum(out.grad*other.item, self_enlarged_dims, keepdims=True)
                    self.grad += self.crop_extra_dims(pre_grad, self)
                if other.requires_grad:
                    pre_grad = np.sum(self.item*out.grad, other_enlarged_dims, keepdims=True)
                    other.grad += self.crop_extra_dims(pre_grad, other)
            # Update _backward function of out                    
            out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * Tensor(-1)

    # - ReLU -
    def relu(self) -> Tensor:
        # Forward
        out = Tensor((self.item > 0)*self.item)
        if self.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, ), requires_grad=True)
            # Backward
            def _backward():
                self.grad += (self.item > 0)*out.grad
            out._backward = _backward
        return out

    # - Reshape -
    def reshape(self, new_shape) -> Tensor:
        # Forward
        out = Tensor(self.item.reshape(new_shape))
        if self.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, ), requires_grad=True)
            # Backward
            def _backward():
                self.grad += out.grad.reshape(self.item.shape)
            out._backward = _backward
        return out

    # - Sum -
    def sum(self, dims_sumed: tuple=None) -> Tensor:
        if dims_sumed == None:
            dims_sumed = tuple(range(len(self.item.shape)))
        # Forward
        out = Tensor(self.item.sum(dims_sumed))
        if self.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, ), requires_grad=True)
            # Backward
            def _backward():
                # Set to 1 the size of input dims where we sum
                mod_in_shape = list(self.item.shape)
                for d in dims_sumed:
                    mod_in_shape[d] = 1
                # Reshape and broadcast
                self.grad += np.broadcast_to(out.grad.reshape(mod_in_shape), self.item.shape)
            out._backward = _backward
        return out

    # - Sigmoid -
    def sigmoid(self) -> Tensor:
        # Forward 
        out = Tensor(1/(1 + np.exp(-self.item)))
        if self.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, ), requires_grad=True)
            # Backward
            def _backward():
                self.grad += (np.exp(self.item)/(1 + np.exp(self.item))**2)*out.grad
            out._backward = _backward
        return out

    # - Softmax -
    def softmax(self, axis: int=-1) -> Tensor:
        # Forward
        max_regulator = np.max(self.item, axis=axis, keepdims=True)
        in_exp = np.exp(self.item - max_regulator)
        norm = np.sum(in_exp, axis=axis, keepdims=True)
        out = Tensor(in_exp/norm)
        if self.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, ), requires_grad=True)
            # Backward
            def _backward():
                # Move axis to the right if not there
                moved_axis = False
                if axis != -1 and axis != len(self.item.shape) - 1:
                    self.item = np.moveaxis(self.item, axis, -1)
                    out.grad = np.moveaxis(out.grad, axis, -1)
                    moved_axis = True
                # Forward objects
                max_regulator = np.max(self.item, axis=-1, keepdims=True)
                in_exp = np.exp(self.item - max_regulator)
                norm = np.sum(in_exp, axis=-1, keepdims=True)
                # Compute gradient
                t1 = in_exp*norm*out.grad
                t2 = in_exp*np.sum(in_exp*out.grad, axis=-1)[..., None]
                if moved_axis:
                    self.grad += np.moveaxis((t1 - t2)/(norm**2), -1, axis)
                else:
                    self.grad += (t1 - t2)/(norm**2)
                # Return axis of self
                if moved_axis:
                    self.item = np.moveaxis(self.item, -1, axis)
            out._backward = _backward
        return out

    # - Divide -
    def __truediv__(self, other: Tensor) -> Tensor:
        self.check_instance(other)
        # Forward
        out = Tensor(self.item/other.item)
        if self.requires_grad or other.requires_grad:
            # Update tensor parameters
            out.__init__(out.item, children=(self, other), requires_grad=True)
            # Backward
            def _backward():
                # Identify broadcasted dims we need to sum over
                self_shape = self.item.shape
                other_shape = other.item.shape
                out_grad_shape = out.grad.shape
                # Enlarge shapes if required
                self_shape = tuple([-1]*(len(out_grad_shape) - len(self_shape))) + self_shape
                other_shape = tuple([-1]*(len(out_grad_shape) - len(other_shape))) + other_shape
                # Find enlarged dims
                self_enlarged_dims, other_enlarged_dims = (), ()
                for dim, (se_d, ot_d, ou_d) in enumerate(zip(self_shape, other_shape, out_grad_shape)):
                    if se_d != ou_d:
                        self_enlarged_dims += (dim, )
                    if ot_d != ou_d:
                        other_enlarged_dims += (dim, )
                if self.requires_grad:
                    pre_grad = np.sum(out.grad/other.item, self_enlarged_dims, keepdims=True)
                    self.grad += self.crop_extra_dims(pre_grad, self)
                if other.requires_grad:
                    pre_grad = np.sum(self.item/(out.grad*out.grad), other_enlarged_dims, keepdims=True)
                    other.grad += self.crop_extra_dims(pre_grad, other)
            # Update _backward function of out                    
            out._backward = _backward
        return out

    #def __rtruediv__(self, other: Tensor) -> Tensor:
    #   return 
        
    # - Backward -
    def backward(self, gradient: np.ndarray=np.array([1])):
        """ gradient is np.array that must have the shape of self.item """
        # Check the gradient dimension is correct
        assert gradient.shape == self.item.shape, \
                "A gradient with the same shape as the item"\
                " must be supplied"
        # Depth-first search of the graph to ensure
        # the gradients are computed in the right order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.requires_grad:
                    for child in v.children:
                        build_topo(child)
                    topo.append(v)
        build_topo(self)
        # Accumulate gradients
        self.grad = gradient
        for tensor in topo[::-1]:
            tensor._backward()