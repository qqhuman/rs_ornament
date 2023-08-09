use std::cell::OnceCell;

pub trait OnceCellExt {
    fn invalidate(&mut self);
}

impl<T> OnceCellExt for OnceCell<T> {
    fn invalidate(&mut self) {
        *self = OnceCell::new();
    }
}
