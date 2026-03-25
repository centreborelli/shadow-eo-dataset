"""
* simplified map interaction using ipyleaflet
* display images in the notebook

Copyright (C) 2017-2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

from __future__ import print_function
import ipywidgets

import folium
import folium.plugins
import numpy as np


def foliummap(location = [48.790153, 2.327395], zoom_start = 13 ):
    """
    creates a folium map centered at the indicated location (lat, long) and zoom
    level indicated by zoom_start. 
    The following widgets are also activated: 
      - drawing polygons and exporting the corresponding geojson 
      - show lat/long of the clicked position   
    Args:
        location (list): list containing lat and long of the center of the map 
        zoom_start (int): zoom level default 13

    Returns:
        handle to the folium.Map object (can be used to add overlays)          
    """
    f = folium.Figure(width='90%')
    m = folium.Map(location=location, zoom_start=zoom_start).add_to(f)
    # we can move the map to any position by indicating its (latitude, longitude)
    #m.location   = location   # i.e. Paris, France
    #m.options['zoom'] = zoom_start

    folium.features.LatLngPopup().add_to(m)
    folium.plugins.Draw(export=True).add_to(m)
    
    return m 



def foliummap_overlay_image(location = [48.790153, 2.327395], zoom_start = 13 ,
                            footprint = None,  imageurl = None):
    """
    creates a folium map centered at the indicated location (lat, long) and zoom
    level indicated by zoom_start, and with an image overlayed at the bounding box of footprint
    The following widgets are also activated: 
      - drawing polygons and exporting the corresponding geojson 
      - show lat/long of the clicked position   
    Args:
        location (list): list containing lat and long of the center of the map 
        zoom_start (int): zoom level default 13
        footprint (GeoJson): polygon delimiting the image to be overlayed
        imageurl (str): location of the image to be overlayed 

    Returns:
        handle to the folium.Map object (can be used to add overlays)          
    """

    """
    creates a folium map with 
    
    """

    m = foliummap(location = location, zoom_start = zoom_start )

    if footprint is not None and imageurl is not None:

        def getbounds_latlon_from_lonlat_polygon(pts):
            """
            Rectangular bounding box for a list of 2D points.

            Args:
                pts (list): list of 2D points represented as 2-tuples or lists of length 2

            Returns:
                ymax, xmax, ymin, xmin (floats): coordinates of the top-left and bottom-right corners
            """
            if type(pts) == list  or  type(pts) == tuple:
                pts = np.array(pts).squeeze()
            dim = len(pts[0])  # should be 2
            bb_min = [min([t[i] for t in pts]) for i in range(dim)]
            bb_max = [max([t[i] for t in pts]) for i in range(dim)]
            return  [[bb_max[1], bb_max[0]], [bb_min[1], bb_min[0]]] 

        # first show the footprint
        folium.GeoJson(footprint).add_to(m)

        # then compute the bounding box of the image footprint  
        bb = getbounds_latlon_from_lonlat_polygon(footprint[ 'coordinates'])
        
        # then add the image overlay at the bounding box location 
        folium.raster_layers.ImageOverlay(image   = imageurl,
                                          bounds  = bb,
                                          opacity = 0.7).add_to(m)
    
    return m 




### simplified map interaction using ipyleaflet

def clickablemap(center = [48.790153, 2.327395], zoom = 13,
                 layout = ipywidgets.Layout(width='90%', height='500px') ):
    # look at: http://leaflet.github.io/Leaflet.draw/docs/examples/basic.html

    import json

    from ipyleaflet import (
        Map,
        Rectangle,
        Polygon,
        TileLayer, ImageOverlay,
        DrawControl, GeoJSON
    )

    #%matplotlib inline
 #   %matplotlib notebook


    # google tileserver 
    # https://stackoverflow.com/questions/9394190/leaflet-map-api-with-google-satellite-layer 
    mosaicsTilesURL = 'https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}' # Hybrid: s,h; Satellite: s; Streets: m; Terrain: p;

    # Create the map
    m = Map(
        center = center,
        zoom = zoom,
        scroll_wheel_zoom = True,
        layout = layout
    )

    # using custom basemap
    m.clear_layers()
    m.add_layer(TileLayer(url=mosaicsTilesURL))

    ## Create the draw controls
    ## @see https://github.com/ellisonbg/ipyleaflet/blob/master/ipyleaflet/leaflet.py#L293
    dc = DrawControl(rectangle={'shapeOptions': {'color': '#0000FF'}})

    m.AOIs = []

    # Register the draw controls handler
    def handle_draw(self, action, geo_json):
        if action != "created":
            return

        m.AOIs.append(geo_json["geometry"])

    # Attach the draw handler to the draw controls `on_draw` event
    dc.on_draw(handle_draw)
    m.add_control(dc)

    # add a custom function to create and add a Rectangle layer
    # (LESS USEFUL THAN add_geojson)
    def add_rect(*args, **kwargs):
        r = Rectangle( *args, **kwargs)
        return m.add_layer(r)
    m.add_rectangle = add_rect

    # add a custom function to create and add a Polygon layer 
    def add_geojson(*args, **kwargs):
        # ugly workaround to call without data=aoi
        if 'data' not in kwargs:
            kwargs['data'] = args[0]
            args2=[i for i in args[1:-1]]
        else:
            args2=args

        r = GeoJSON( *args2, **kwargs)
        return m.add_layer(r)
    m.add_GeoJSON = add_geojson

    # Display
    return m


def overlaymap(aoiY, imagesurls, zoom = 13,
               layout = ipywidgets.Layout(width='90%', height='500px') ):
    
    import json
    import numpy as np
    
    from ipyleaflet import (
        Map,
        Rectangle,
        Polygon,
        TileLayer, ImageOverlay,
        DrawControl, 
    )

    ## handle the case of imageurls not a list
    if type(imagesurls) != list:
        imagesurls = [imagesurls]
        
    number_of_images = len(imagesurls)
    
    ## handle both kinds of calls with aoi, or aoi['coordinates']
    if 'coordinates' in aoiY:
        aoiY=aoiY['coordinates'][0]
        
        
    # create the Map object    
    # google tileserver 
    # https://stackoverflow.com/questions/9394190/leaflet-map-api-with-google-satellite-layer 
    mosaicsTilesURL = 'https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}' # Hybrid: s,h; Satellite: s; Streets: m; Terrain: p;
    m = Map( center = aoiY[0][::-1] , 
            zoom = zoom,
            scroll_wheel_zoom = True,
            layout = layout,
       )

    # using custom basemap 
    m.clear_layers()
    m.add_layer(TileLayer(url=mosaicsTilesURL, opacity=1.00))
    
    #vlayer = VideoOverlay(videoUrl, videoBounds )
    #m.add_layer(vlayer)



    ### this shows an animated gif
    #m.add_layer(layer)


    # display map (this show)
    #display(m)



    ############## ADD INTERACTIVE LAYER
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets

    
    # meke sure that the images have unique names 
    imagesurls =  ['%s?%05d'%(i,np.random.randint(10000)) for i in imagesurls] 

    
    # draw bounding polygon
    y = [ a[::-1] for a in aoiY ]
    p = Polygon(locations=y, weight=2, fill_opacity=0.25)
    m.add_layer(p)

    # create image 
    layer = ImageOverlay(url='%s'%(imagesurls[0]), bounds=[ list(np.max(aoiY,axis=0)[::-1]) , list(np.min(aoiY,axis=0)[::-1]) ])

    m.add_layer(layer)

    # callback fro flipping images
    def showim(i):
            if(i<len(imagesurls)):
   #      -----       FLICKERS ----
   #             layer.url='%s'%(imagesurls[i])
   #             layer.visible = False
   #             layer.visible = True
    
   #    ALTERNATIVE:  add a new layer 
                layer = ImageOverlay(url='%s'%(imagesurls[i]), bounds=[ list(np.max(aoiY,axis=0)[::-1]) , list(np.min(aoiY,axis=0)[::-1]) ])
                m.add_layer(layer)
                # remove old ones 
                if len(m.layers)>30: # image buffer 
                    for l in (m.layers[1:-1]):
                        m.remove_layer(l)
    
    
    # build the UI
    #interact(showim,i=len(imagesurls)-1)
        #interact(showim, i=widgets.IntSlider(min=0,max=len(imagesurls),step=1,value=0));
    play = widgets.Play(
        interval=200,   #ms
        value=0,
        min=0,
        max=len(imagesurls)-1,
        step=1,
        description="Press play",
        disabled=False,
    )
    slider = widgets.IntSlider( min=0, max=len(imagesurls)-1, description='Frame:')
    label  = widgets.Label(value="")
    def on_value_change(change):
        label.value=imagesurls[change['new']] 
        showim(change['new'])
    slider.observe(on_value_change, 'value')
    b1 = widgets.Button(description='fw', layout=widgets.Layout(width='auto') )
    b2 = widgets.Button(description='bw', layout=widgets.Layout(width='auto'))
    b3 = widgets.Button(description='hide', layout=widgets.Layout(width='auto'))
    b4 = widgets.Button(description='hidePoly', layout=widgets.Layout(width='auto'))
    def clickfw(b):
            slider.value=slider.value+1
    def clickbw(b):
        slider.value=slider.value-1
    def clickhide(b):
        if layer.visible:
            layer.visible = False
        else:
            layer.visible = True
    def clickhidePoly(b):
        if p.fill_opacity>0:
            p.fill_opacity = 0
            p.weight=0
        else:
            p.fill_opacity = 0.25
            p.weight=2
    b1.on_click( clickfw )
    b2.on_click( clickbw )
    b3.on_click( clickhide )
    b4.on_click( clickhidePoly )

    
    # add a custom function to create and add a Polygon layer 
    def add_geojson(*args, **kwargs):
        # ugly workaround to call without data=aoi
        if 'data' not in kwargs:
            kwargs['data'] = args[0]
            args2=[i for i in args[1:-1]]
        else:
            args2=args

        r = GeoJSON( *args2, **kwargs)
        return m.add_layer(r)
    m.add_GeoJSON = add_geojson 
    

    widgets.jslink((play, 'value'), (slider, 'value'))
    if number_of_images>1:
        return widgets.VBox([widgets.HBox([play,b2,b1,b3,b4, slider,label]),m])
    else:
        return widgets.VBox([widgets.HBox([b3,b4, label]),m])        
    #interactive(showim, i=slider   )
       

        

### DISPLAY IMAGES AND TABLES IN THE NOTEBOOK


# utility function for printing with Markdown format
def printmd(string):
    from IPython.display import Markdown, display
    display(Markdown(string))

    
def printbf(obj):
    printmd("__"+str(obj)+"__")

    
def show_array(a, fmt='jpeg'):
    ''' 
    display a numpy array as an image
    supports monochrome (shape = (N,M,1) or (N,M))
    and color arrays (N,M,3)  
    '''
    import PIL.Image
    from io import BytesIO
    import IPython.display
    import numpy as np
        
    #handle color images (3,N,M) -> (N,M,3)
    a = a.squeeze()
    if len(a.shape) == 3 and a.shape[0] == 3:
        a = a.transpose(1,2,0)
        
    f = BytesIO()
    PIL.Image.fromarray(np.uint8(a).squeeze() ).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def display_image(img):
    '''
    display_image(img)
    display an image in the curren IPython notebook
    img can be an url, a local path, a numpy array, of a PIL image 
    '''
    from IPython.display import display, Image
    import PIL.PngImagePlugin
    from urllib import parse   
    import numpy as np
    
    if type(img) == np.ndarray:
        x = np.squeeze(img).copy()
        show_array(x)
    elif type(img) == PIL.PngImagePlugin.PngImageFile:
        display(img)
    elif parse.urlparse(img).scheme in ('http', 'https', 'ftp'):
        display(Image(url=img)) 
    else:
        # read the encoded image and add it to the notebook
        # format='png' is just needed to correctly display gifs: https://github.com/ipython/ipython/issues/10045
        with open(img,'rb') as f:
            display(Image(data=f.read(), format='png'))
        # local url cannot be displayed in colab
        #display(Image(filename=img)) 


    

def display_imshow(im, range=None, cmap='gray', axis='equal', invert=False,
                   title=None, inline=False, show=True, figsize=(7,5)):
    """
    Display an image using matplotlib.pyplot.imshow().

    Args:
        im (str or np.array): can be an url, a local path, or a numpy array
        range (list): list of length two with vmin, vmax
        cmap (str): set the colormap ('gray', 'jet', ...)
        axis (str): set the scale of the axis ('auto', 'equal', 'off')
            see https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.axis.html
        invert (bool): reverses the y-axis
        title (str): set the figure title
        inline (bool): forces inline when jupyter is in notebook mode
        figsize ( default (5,5) ): force figure size   
        show (True): ends by showing the figure, use false to overlay
    """
    import matplotlib.pyplot as plt
    import matplotlib 
    
    
    # mute the interactive mode in order to put the figure inline
    wasinteractive = matplotlib.is_interactive()
    if inline:
        plt.ioff()

        
    vmin, vmax = None, None
    if range:
        vmin, vmax = range[0], range[1]
    plt.figure(figsize=figsize)  # figsize=(13, 10)

    # handle color images
    im = im.squeeze()
    if len(im.shape) == 3 and im.shape[0] == 3:
        im = im.transpose(1, 2, 0)

    plt.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    if title:
        plt.title(title)
    if invert:
        plt.gca().invert_yaxis()
    plt.axis(axis)
    plt.colorbar()
    
    # handle the inline case
    if inline: 
        import PIL.Image
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = PIL.Image.open(buf)        
        display_image(im)
        
        # close figure and buffer
        buf.close()
        plt.close()
        
        # unmute the interactive mode 
        if wasinteractive:
            plt.ion() 
    
    if show:
        plt.show()

def urlencoded_jpeg_img(a):
    ''' 
    returns the string of an html img tag with the urlencoded jpeg of 'a'
    supports monochrome (shape = (N,M,1) or (N,M))
    and color arrays (N,M,3)  
    '''
    fmt='jpeg'
    import PIL.Image
    from io import BytesIO
    import IPython.display
    import numpy as np
    import base64
        
    #handle color images (3,N,M) -> (N,M,3)
    a = a.squeeze()
    if len(a.shape) == 3 and a.shape[0] == 3:
        a = a.transpose(1,2,0)

    f = BytesIO()
    PIL.Image.fromarray(np.uint8(a).squeeze() ).save(f, fmt)
    x =  base64.b64encode(f.getvalue())
    return '''<img src="data:image/jpeg;base64,{}&#10;"/>'''.format(x.decode())
    # display using IPython.display.HTML(retval)
    
       
### initialize gallery
        
gallery_style_base = """
    <style>
.gallery2 {
    position: relative;
    width: auto;
    height: 650px; }
.gallery2 .index {
    padding: 0;
    margin: 0;
    width: 10.5em;
    list-style: none; }
.gallery2 .index li {
    margin: 0;
    padding: 0;
    float: left;}
.gallery2 .index a { /* gallery2 item title */
    display: block;
    background-color: #EEEEEE;
    border: 1px solid #FFFFFF;
    text-decoration: none;
    width: 1.9em;
    padding: 6px; }
.gallery2 .index a span { /* gallery2 item content */
    display: block;
    position: absolute;
    left: -9999px; /* hidden */
    top: 0em;
    padding-left: 0em; }
.gallery2 .index a span img{ /* gallery2 item content */
    height: 550px;
    }
.gallery2 .index li:first-child a span {
    top: 0em;
    left: 10.5em;
    z-index: 99; }
.gallery2 .index a:hover {
    border: 1px solid #888888; }
.gallery2 .index a:hover span {
    left: 10.5em;
    z-index: 100; }
</style>
    """

svg_overlay_style = """
<style>
.svg-overlay {
  position: relative;
  display: inline-block;
}

.svg-overlay svg {
  position: absolute;
  top: 0;
  left: 0;
}
</style>
"""
  
def display_gallery(image_urls, image_labels=None, svg_overlays=None):
    '''
    image_urls can be a list of urls 
    or a list of numpy arrays
    image_labels is a list of strings
    '''
    from  IPython.display import HTML 
    import numpy as np
    from urllib import parse   
    import PIL.Image

    
    gallery_template = """
    <div class="gallery2">
        <ul class="index">
            {}
        </ul>
    </div>
    """
    
    li_template = """<li><a href="#">{}<span style="background-color: white;  " ><img src="{}" /></br>{}</span></a></li>"""
    li_template_encoded = """<li><a href="#">{}<span style="background-color: white;  " >{}</br>{}</span></a></li>"""

    li = ""
    idx = 0
    for u in image_urls:
        if image_labels:
            label = image_labels[idx]
        else:
            label = str(idx)

        if svg_overlays:
            svg = svg_overlays[idx]
        else:
            svg = None

        if type(u) == str and parse.urlparse(u).scheme in ('http', 'https', 'ftp'):  # full url
            li = li + li_template.format( idx, u, label)
        elif type(u) == str:   # assume relative url path
            img = np.asarray(PIL.Image.open(u))
            li = li + li_template_encoded.format( idx, urlencoded_jpeg_img(img), label)                
        elif type(u) == np.ndarray:   # input is np.array
            h, w = u.shape[0], u.shape[1]
            div = f'<div class="svg-overlay">{urlencoded_jpeg_img(u)}<svg viewBox="0 0 {w} {h}">{svg}</svg></div>'
            li = li + li_template_encoded.format(idx, div, label)

        idx = idx + 1
        
    source = gallery_template.format(li)
    
    display(HTML( source ))
    display(HTML( gallery_style_base ))
    display(HTML(svg_overlay_style))

    return 
    


def overprintText(im,imout,text,textRGBA=(255,255,255,255)):
    '''
    prints text in the upper left corner of im (filename) 
    and writes imout (filename)
    '''
    from PIL import Image, ImageDraw, ImageFont
    # get an image
    base = Image.open(im).convert('RGBA')

    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255,255,255,0))

    # get a font
    #    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(txt)

    # draw text
    d.text((1,1), text,  fill=tuple(textRGBA))
    out = Image.alpha_composite(base, txt)

    out.save(imout)




def make_animated_gif(out_filename, files, delay=500, loop=0):
    '''
    generate an animated gif from files
    '''
    import glob
    from PIL import Image, ImageDraw
    images = []
    for f in  glob.glob(files):
        im = Image.open(f)
        images.append(im)
    images[0].save(out_filename,
       save_all=True, append_images=images[1:], optimize=False, duration=delay, loop=loop)
        



