import numpy as np
import tensorflow as tf


class WindowGenerator:
    def __init__(self, input_width, target_width, offset, ts, target_columns=None):
        # Store the raw data.
        self.ts = ts
        self.df = ts.pd_dataframe()

        # Work out the target column indices.
        self.target_columns = target_columns
        if target_columns is not None:
            self.target_columns_indices = {name: i for i, name in
                                           enumerate(target_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.target_width = target_width
        self.offset = offset

        self.total_window_size = input_width + offset

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.target_start = self.total_window_size - self.target_width
        self.targets_slice = slice(self.target_start, None)
        self.target_indices = np.arange(self.total_window_size)[self.targets_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Target indices: {self.target_indices}',
            f'Target column name(s): {self.target_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        targets = features[:, self.targets_slice, :]
        if self.target_columns is not None:
            targets = tf.stack(
                [targets[:, :, self.column_indices[name]] for name in self.target_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        targets.set_shape([None, self.target_width, None])

        return inputs, targets

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        return ds.map(self.split_window)

    @property
    def dataset(self):
        return self.make_dataset(self.df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, targets` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the dataset
            result = next(iter(self.dataset))
            # And cache it for next time
            self._example = result
        return result

