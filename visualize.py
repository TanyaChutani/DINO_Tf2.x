import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize

image_size = 224

url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
image = utils.read(url, image_size)
attention_map = visualize.attention_map(model=vittt, image=image)

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_title('Original')
ax2.set_title('Attention Map')
_ = ax1.imshow(image)
_ = ax2.imshow(attention_map)
