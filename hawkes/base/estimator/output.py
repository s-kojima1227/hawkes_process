import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import japanize_matplotlib
from ..events import Events

class Output:
    def __init__(self, events: Events, t, intensity, params, kernel_type, loglik):
        self._events = events.grouped_by_mark
        self._T = events.end_time
        self._dim = events.dim
        self._t = t
        self._intensity = intensity
        self._params = params
        self._kernel_type = kernel_type
        self._loglik = loglik

    @property
    def events(self):
        if (self._dim == 1):
            return self._events[0]
        return self._events

    @property
    def T(self):
        return self._T

    @property
    def intensity(self):
        if (self._dim == 1):
            return self._intensity[0]
        return self._intensity

    @property
    def params(self):
        return self._params

    @property
    def kernel_type(self):
        return self._kernel_type

    @property
    def loglik(self):
        return self._loglik

    def info(self):
        kernel_type_text = '- kernel_type: {}'.format(self.kernel_type)
        params_text = '- params: {}'.format(self._params)
        end_time_text = '- end_time: {}'.format(self._T)
        loglik_text = '- loglik: {}'.format(self._loglik)
        events_text = '- events:\n' + '\n'.join(['  - dim_{}: {}'.format(i + 1, np.round(self._events[i], 2)) for i in range(self._dim)])
        text = '\n'.join([kernel_type_text, params_text, end_time_text, loglik_text, events_text])
        print(text)

    def plot(self):
        fig, (ax_legend, ax1, ax2, ax3) = plt.subplots(4, 1, sharex=True, figsize=(20, 5))
        plt.subplots_adjust(hspace=0.4)
        padding = 1
        ax1.set_xlim(0 - padding, self._T + padding)
        color_palette = plt.cm.tab10
        ax_legend.axis('off')
        handles = []  # 凡例のためのハンドルを格納するリスト

        for i in range(self._dim):
            # カラーパレットから色を選択
            color = color_palette(i)
            handles.append(mpatches.Patch(color=color, label=f'次元 {i+1}'))

            events_i = self._events[i]
            intensity_i = self._intensity[i]

            event_with_bounds = np.hstack([0, events_i, self._T])
            cumulative_counts = np.arange(0, len(events_i) + 1)
            cumulative_counts = np.hstack([cumulative_counts, cumulative_counts[-1]])

            # 累積イベント数をプロット
            ax1.step(event_with_bounds, cumulative_counts, where='post', label=f'次元 {i+1}', color=color)
            ax1.set_title('累積イベント数')

            # イベント発生時刻をプロット
            ax2.vlines(events_i, ymin=0, ymax=1, linestyles='solid', label='イベント発生', color=color)
            ax2.tick_params(left=False, labelleft=False)
            ax2.set_title('イベント発生時刻')

            # 条件付き強度をプロット
            ax3.plot(self._t, intensity_i, color=color)
            ax3.set_title('強度')

        ax_legend.legend(handles=handles, loc='lower left', ncol=self._dim)
        plt.show()