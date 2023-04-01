class RAdam(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        p_max: float = 4.0,
        inplace: bool = True,
        stop_gradients: bool = True,
        compile_on_next_step: bool = False,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ):
        """
        Construct an RAdam optimizer.
        Parameters
        ----------
        lr
            Learning rate, default is ``1e-3``.
        beta1
            gradient forgetting factor, default is ``0.9``
        beta2
            second moment of gradient forgetting factor, default is ``0.999``
        epsilon
            divisor during adam update, preventing division by zero,
            default is ``1e-8``
        p_max
            maximum bound for the moving average of the variance, default is ``4.0``
        inplace
            Whether to update the variables in-place, or to create new variable handles.
            This is only relevant for frameworks with stateful variables such as
            PyTorch.
            Default is ``True``, provided the backend framework supports it.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is ``True``.
        compile_on_next_step
            Whether to compile the optimizer on the next step. Default is ``False``.
        device
            Device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. (Default value = None)
        """
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
	    self._p_max = _p_max
        self._mw = None
        self._vw = None
        self._first_pass = True
        self._should_compile = False


        Optimizer.__init__(
            self,
            lr,
            inplace,
            stop_gradients,
            compile_on_next_step,
            device=device
        )

    # Custom Step

    def _step(self, v: ivy.Container, grads: ivy.Container):
        """
        Update nested variables container v by RAdam update step,
        using nested grads container.
        Parameters
        ----------
        v
            Nested variables to update.
        grads
            Nested gradients to update.
        Returns
        -------
        ret
            The updated variables, following RAdam update step.
        """
        if self._first_pass:
            self._mw = grads
            self._vw = grads**2   
            self._first_pass = False
        new_v, self._mw, self._vw, self._p_max = ivy.radam_update(
            v,
            grads,
            self._lr if isinstance(self._lr, float) else self._lr(),
            self._mw,
            self._vw,
            self._count,
            beta1=self._beta1,
            beta2=self._beta2,
            epsilon=self._epsilon,
	    p_max=self._p_max,
            stop_gradients=self._stop_gradients,
        )
        return new_v

    def set_state(self, state: ivy.Container):
        """
        Set state of the optimizer.
        Parameters
        ----------
        state
            Nested state to update.
        """
        self._mw = state.mw
        self._vw = state.vw

    @property
    def state(self):
        return ivy.Container({"mw": self._mw, "vw": self._vw})

def radam_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    mw_tm1: Union[ivy.Array, ivy.NativeArray],
    vw_tm1: Union[ivy.Array, ivy.NativeArray],
    step: int,
    p_max: float,
    /,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-7,
    stop_gradients: bool = True,
    out: Optional[ivy.Array] = None,
) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:

    """Update weights w of some function, given the derivatives of some cost c with
    respect to w, using RAdam update.

    Parameters
    ----------
    w
        Weights of the function to be updated.
    dcdw
        Derivatives of the cost c with respect to the weights w, [dc/dw for w in ws].
    lr
        Learning rate(s), the rate(s) at which the weights should be updated relative to
        the gradient.
    mw_tm1
        Running average of the gradients, from the previous time-step.
    vw_tm1
        Running average of second moments of the gradients, from the previous time-step.
    step
        Training step.
    p_max
        The maximum value of the trust ratio.

    beta1
        Gradient forgetting factor (Default value = 0.9).
    beta2
        Second moment of gradient forgetting factor (Default value = 0.999).
    epsilon
        Divisor during adam update, preventing division by zero (Default value = 1e-7).
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is ``True``.
    out
        Optional output array, for writing the new function weights w_new to. It must
        have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The new function weights w_new, and also new mw and vw, following the radam
        updates.

    """
    effective_grads, mw, vw = ivy.radam_step(
        dcdw, mw_tm1, vw_tm1, step, beta1=beta1, beta2=beta2, epsilon=epsilon, p_max=p_max
    )
 
def radam_step(
    dcdw: Union[ivy.Array, ivy.NativeArray],
    mw: Union[ivy.Array, ivy.NativeArray],
    vw: Union[ivy.Array, ivy.NativeArray],
    step: Union[int, float],
    /,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-7,
    p_max: float = 4.0,
    out: Optional[ivy.Array] = None,
) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
    step = float(step)
    t=t+1
    mw = beta1 * mw + (1 - beta1) * dcdw
    vw = 1 / beta2 * vw + (1 - beta2) * np.power(dcdw, 2)

    m_hat = mw / (1 - beta1**t)
    p_t = p_max - 2 *t * beta2**t / (1 - beta2**t)

    if p_t > 4:
        l_t = np.sqrt((1 - beta2**t) / vw)
        r_t = np.sqrt(((p_t - 4) * (p_t - 2) * p_max) / ((p_max - 4) * (p_max - 2) * p_t))
        w_update = step * r_t * m_hat * l_t
    else:
        w_update = step * m_hat

    return w - w_update