import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np
import japanize_matplotlib
from scipy.ndimage import rotate

def draw_on_image():
    # 画像を読み込み背景として表示（ピクセル座標に合わせてextentを設定）
    img = mpimg.imread('../syuto.png')
    rotation_angle = -0.4  # 画像を少し回転（度数法）
    img = rotate(img, rotation_angle, reshape=False, mode='nearest')
    fig, ax = plt.subplots(figsize=(10, 8))
    img_height, img_width = img.shape[:2]
    ax.imshow(img, extent=(0, img_width, img_height, 0), origin='upper', zorder=0)

    # ▼ここで画像上の特定領域にグラフ用Axesを作成する▼
    # 例: ピクセル座標で (x0=400, y0=200) を左上に、幅=600、高さ=350 の範囲
    x0, y0 = -90, 100          # 画像上でグラフを始める位置（左上）
    box_width, box_height = 2500, 920  # グラフ領域の幅と高さ

    # 画像座標の Bbox を作って、Figure 座標に変換
    bbox_data = Bbox.from_bounds(x0, y0, box_width, box_height)
    bbox_disp = TransformedBbox(bbox_data, ax.transData)                # データ座標 → 画面座標
    bbox_fig  = TransformedBbox(bbox_disp, fig.transFigure.inverted())  # 画面座標 → Figure 座標

    # 指定した領域に新しい Axes（小窓）を追加
    sub_ax = fig.add_axes(bbox_fig.bounds, zorder=1)
    sub_ax.set_facecolor('none')  # 背景を透過
    sub_ax.patch.set_alpha(0.0)

    # 小窓にグラフを描く
    sub_ax.grid(True)

    # 元の全体軸（背景用）に軸目盛り・枠が不要なら消す
    ax.set_axis_off()
    plt.tight_layout()
    return fig
