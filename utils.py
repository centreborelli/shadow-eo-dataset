"""
* GeoTIFF read and write
* extract metadata: datetime, RPC,
* miscelaneous functions for crop
* wrappers for gdaltransform and gdalwarp

Copyright (C) 2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Copyright (C) 2018, Carlo de Franchis <carlo.de-franchis@ens-paris-saclay.fr>
"""

import datetime
import json
import math
import os
import subprocess
import warnings
from itertools import combinations

import bs4
import geojson
import numpy as np
import pyproj
import rasterio
import requests
import rpcm
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def readGTIFF(fname):
    """
    Reads an image file into a numpy array,
    returns the numpy array with dimensios (height, width, channels)
    The returned numpy array is always of type numpy.float
    """
    # read the image into a np.array
    with rasterio.open(fname, "r") as s:
        # print('reading image of size: %s'%str(im.shape))
        im = s.read()
    return im.transpose([1, 2, 0]).astype(float)


def readGTIFFmeta(fname):
    """
    Reads the image GeoTIFF metadata using rasterio and returns it,
    along with the bounding box, in a tuple: (meta, bounds)
    if the file format doesn't support metadata the returned metadata is invalid
    This is the metadata rasterio was capable to interpret,
    but the ultimate command for reading metadata is *gdalinfo*
    """
    with rasterio.open(fname, "r") as s:
        ## interesting information
        # print(s.crs,s.meta,s.bounds)
        return (s.meta, s.bounds)


def get_driver_from_extension(filename):
    import os.path

    ext = os.path.splitext(filename)[1].upper()
    if ext in (".TIF", ".TIFF"):
        return "GTiff"
    elif ext in (".JPG", ".JPEG"):
        return "JPEG"
    elif ext == ".PNG":
        return "PNG"
    return None


def writeGTIFF(im, fname, copy_metadata_from=None):
    """
    Writes a numpy array to a GeoTIFF, PNG, or JPEG image depending on fname extension.
    For GeoTIFF files the metadata can be copied from another file.
    Note that if  im  and  copy_metadata_from have different size,
    the copied geolocation properties are not adapted.
    """
    import numpy as np
    import rasterio

    # set default metadata profile
    p = {
        "width": 0,
        "height": 0,
        "count": 1,
        "dtype": "uint8",
        "driver": "PNG",
        "affine": rasterio.Affine(0, 1, 0, 0, 1, 0),
        "crs": rasterio.crs.CRS({"init": "epsg:32610"}),
        "tiled": False,
        "nodata": None,
    }

    # read and update input metadata if available
    if copy_metadata_from:
        x = rasterio.open(copy_metadata_from, "r")
        p.update(x.profile)

    # format input
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]

    # override driver and shape
    indriver = get_driver_from_extension(fname)
    if indriver and (indriver != p["driver"]):
        # print('writeGTIFF: driver override from %s to %s'%( p['driver'], indriver))
        p["driver"] = indriver or p["driver"]
        p["dtype"] = "float32"

    # if indriver == 'GTiff' and (p['height'] != im.shape[0]  or  p['width'] != im.shape[1]):
    #    # this is a problem only for GTiff
    #    print('writeGTIFF: changing the size of the GeoTIFF')
    # else:
    #    # remove useless properties
    #    p.pop('tiled')

    p["height"] = im.shape[0]
    p["width"] = im.shape[1]
    p["count"] = im.shape[2]

    with rasterio.open(fname, "w", **p) as d:
        d.write((im.transpose([2, 0, 1]).astype(d.profile["dtype"])))


def is_absolute(url):
    return bool(requests.utils.urlparse(url).netloc)


def find(url, extension):
    """
    Recursive directory listing, like "find . -name "*extension".

    Args:
        url (str): directory url
        extension (str): file extension to match

    Returns:
        list of urls to files
    """
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = bs4.BeautifulSoup(r.text, "html.parser")
    files = [
        node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith(extension)
    ]
    folders = [
        node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith("/")
    ]

    files_urls = [
        f if is_absolute(f) else os.path.join(url, os.path.basename(f)) for f in files
    ]
    folders_urls = [
        f if is_absolute(f) else os.path.join(url, os.path.basename(f.rstrip("/")))
        for f in folders
    ]

    for u in folders_urls:
        if not u.endswith(("../", "..")):
            files_urls += find(u, extension)
    return files_urls


def acquisition_date(geotiff_path):
    """
    Read the image acquisition date in GeoTIFF metadata.

    Args:
        geotiff_path (str): path or url to a GeoTIFF file

    Returns:
        datetime.datetime object with the image acquisition date
    """
    with rasterio.open(geotiff_path, "r") as src:
        if "NITF_IDATIM" in src.tags():
            date_string = src.tags()["NITF_IDATIM"]
        elif "NITF_STDIDC_ACQUISITION_DATE" in src.tags():
            date_string = src.tags()["NITF_STDIDC_ACQUISITION_DATE"]
        return datetime.datetime.strptime(date_string, "%Y%m%d%H%M%S")


def gdal_get_longlat_of_pixel(fname, x, y, verbose=True):
    """
    returns the longitude latitude and altitude (wrt the WGS84 reference
    ellipsoid) for the points at pixel coordinates (x, y) of the image fname.
    The CRS of the input GeoTIFF is determined from the metadata in the file.

    """
    import subprocess

    # add vsicurl prefix if needed
    env = os.environ.copy()
    if fname.startswith(("http://", "https://")):
        env["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = fname[-3:]
        fname = "/vsicurl/{}".format(fname)

    # form the query string for gdaltransform
    q = b""
    for xi, yi in zip(x, y):
        q = q + b"%d %d\n" % (xi, yi)
    # call gdaltransform, "+proj=longlat" uses the WGS84 ellipsoid
    #    echo '0 0' | gdaltransform -t_srs "+proj=longlat" inputimage.tif
    cmdlist = ["gdaltransform", "-t_srs", "+proj=longlat", fname]
    if verbose:
        print("RUN: " + " ".join(cmdlist) + " [x y from stdin]")
    p = subprocess.Popen(cmdlist, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = (p.communicate(q)[0]).decode()
    listeout = [list(map(float, x.split())) for x in out.splitlines()]
    return listeout


def lon_lat_image_footprint(image, z=0):
    """
    Compute the longitude, latitude footprint of an image using its RPC model.

    Args:
        image (str): path or url to a GeoTIFF file
        z (float): altitude (in meters above the WGS84 ellipsoid) used to
            convert the image corners pixel coordinates into longitude, latitude

    Returns:
        geojson.Polygon object containing the image footprint polygon
    """
    rpc = rpcm.rpc_from_geotiff(image)
    with rasterio.open(image, "r") as src:
        h, w = src.shape
    coords = []
    for x, y, z in zip([0, w, w, 0], [0, 0, h, h], [z, z, z, z]):
        lon, lat = rpc.localization(x, y, z)
        coords.append([lon, lat])
    return geojson.Polygon([coords])


def gdal_resample_image_to_longlat(fname, outfname, verbose=True):
    """
    resample a geotiff image file in longlat coordinates (EPSG: 4326 with WGS84 datum)
    and saves the result in outfname
    """
    import os

    driver = get_driver_from_extension(outfname)
    cmd = 'gdalwarp -overwrite  -of %s -t_srs "+proj=longlat +datum=WGS84" %s %s' % (
        driver,
        fname,
        outfname,
    )
    if verbose:
        print("RUN: " + cmd)
    return os.system(cmd)


def bounding_box2D(pts):
    """
    Rectangular bounding box for a list of 2D points.

    Args:
        pts (list): list of 2D points represented as 2-tuples or lists of length 2

    Returns:
        x, y, w, h (floats): coordinates of the top-left corner, width and
            height of the bounding box
    """
    if type(pts) == list or type(pts) == tuple:
        pts = np.array(pts).squeeze()
    dim = len(pts[0])  # should be 2
    bb_min = [min([t[i] for t in pts]) for i in range(dim)]
    bb_max = [max([t[i] for t in pts]) for i in range(dim)]
    return bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]


def image_crop_gdal(inpath, x, y, w, h, outpath):
    """
    Image crop defined in pixel coordinates using gdal_translate.

    Args:
        inpath: path to an image file
        x, y, w, h: four integers defining the rectangular crop pixel coordinates.
            (x, y) is the top-left corner, and (w, h) are the dimensions of the
            rectangle.
        outpath: path to the output crop
    """
    if int(x) != x or int(y) != y:
        print("WARNING: image_crop_gdal will round the coordinates of your crop")

    env = os.environ.copy()
    if inpath.startswith(("http://", "https://")):
        env["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = inpath[-3:]
        path = "/vsicurl/{}".format(inpath)
    else:
        path = inpath

    cmd = [
        "gdal_translate",
        path,
        outpath,
        "-srcwin",
        str(x),
        str(y),
        str(w),
        str(h),
        "-ot",
        "Float32",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=IF_NEEDED",
    ]

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        if inpath.startswith(("http://", "https://")):
            if not requests.head(inpath, timeout=30).ok:
                print("{} is not available".format(inpath))
                return
        print("ERROR: this command failed")
        print(" ".join(cmd))
        print(e.output)


def points_apply_homography(H, pts):
    """
    Applies an homography to a list of 2D points.

    Args:
        H (np.array): 3x3 homography matrix
        pts (list): list of 2D points, each point being a 2-tuple or a list
            with its x, y coordinates

    Returns:
        a numpy array containing the list of transformed points, one per line
    """
    pts = np.asarray(pts)

    # convert the input points to homogeneous coordinates
    if len(pts[0]) < 2:
        print(
            """points_apply_homography: ERROR the input must be a numpy array
          of 2D points, one point per line"""
        )
        return
    pts = np.hstack((pts[:, 0:2], pts[:, 0:1] * 0 + 1))

    # apply the transformation
    Hpts = (np.dot(H, pts.T)).T

    # normalize the homogeneous result and trim the extra dimension
    Hpts = Hpts * (1.0 / np.tile(Hpts[:, 2], (3, 1))).T
    return Hpts[:, 0:2]


def bounding_box_of_projected_aoi(rpc, aoi, z=0, homography=None):
    """
    Return the x, y, w, h pixel bounding box of a projected AOI.

    Args:
        rpc (rpcm.RPCModel): RPC camera model
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI
        z (float): altitude of the AOI with respect to the WGS84 ellipsoid
        homography (2D array, optional): matrix of shape (3, 3) representing an
            homography to be applied to the projected points before computing
            their bounding box.

    Return:
        x, y (ints): pixel coordinates of the top-left corner of the bounding box
        w, h (ints): pixel dimensions of the bounding box
    """
    lons, lats = np.array(aoi["coordinates"][0]).T
    x, y = rpc.projection(lons, lats, z)
    pts = list(zip(x, y))
    if homography is not None:
        pts = points_apply_homography(homography, pts)
    return np.round(bounding_box2D(pts)).astype(int)


def crop_aoi(geotiff, aoi, z=0):
    """
    Crop a geographic AOI in a georeferenced image using its RPC functions.

    Args:
        geotiff (string): path or url to the input GeoTIFF image file
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI
        z (float, optional): base altitude with respect to WGS84 ellipsoid (0
            by default)

    Return:
        crop (array): numpy array containing the cropped image
        x, y, w, h (ints): image coordinates of the crop. x, y are the
            coordinates of the top-left corner, while w, h are the dimensions
            of the crop.
    """
    x, y, w, h = bounding_box_of_projected_aoi(rpcm.rpc_from_geotiff(geotiff), aoi, z)
    with rasterio.open(geotiff, "r") as src:
        crop = src.read(window=((y, y + h), (x, x + w)), boundless=True).squeeze()
    return crop, x, y


def lonlat_to_utm(lons, lats, force_epsg=None):
    """
    Convert longitude, latitude to UTM coordinates.

    Args:
        lons (float or list): longitude, or list of longitudes
        lats (float or list): latitude, or list of latitudes
        force_epsg (int): optional EPSG code of the desired UTM zone

    Returns:
        eastings (float or list): UTM easting coordinate(s)
        northings (float or list): UTM northing coordinate(s)
        epsg (int): EPSG code of the UTM zone
    """
    lons = np.atleast_1d(lons)
    lats = np.atleast_1d(lats)
    if force_epsg:
        epsg = force_epsg
    else:
        epsg = compute_epsg(lons[0], lats[0])
    e, n = pyproj_lonlat_to_epsg(lons, lats, epsg)
    return e.squeeze(), n.squeeze(), epsg


def utm_to_lonlat(eastings, northings, epsg):
    """
    Convert UTM coordinates to longitude, latitude.

    Args:
        eastings (float or list): UTM easting coordinate(s)
        northings (float or list): UTM northing coordinate(s)
        epsg (int): EPSG code of the UTM zone

    Returns:
        lons (float or list): longitude, or list of longitudes
        lats (float or list): latitude, or list of latitudes
    """
    eastings = np.atleast_1d(eastings)
    northings = np.atleast_1d(northings)
    lons, lats = pyproj_epsg_to_lonlat(eastings, northings, epsg)
    return lons.squeeze(), lats.squeeze()


def utm_zone_to_epsg(utm_zone, northern_hemisphere=True):
    """
    Args:
        utm_zone (int):
        northern_hemisphere (bool): True if northern, False if southern

    Returns:
        epsg (int): epsg code
    """
    # EPSG = CONST + ZONE where CONST is
    # - 32600 for positive latitudes
    # - 32700 for negative latitudes
    const = 32600 if northern_hemisphere else 32700
    return const + utm_zone


def epsg_to_utm_zone(epsg):
    """
    Args:
        epsg (int):

    Returns:
        utm_zone (int): zone number
        northern_hemisphere (bool): True if northern, False if southern
    """
    if 32600 < epsg <= 32660:
        return epsg % 100, True
    elif 32700 < epsg <= 32760:
        return epsg % 100, False
    else:
        raise Exception("Invalid UTM epsg code: {}".format(epsg))


def compute_epsg(lon, lat):
    """
    Compute the EPSG code of the UTM zone which contains
    the point with given longitude and latitude

    Args:
        lon (float): longitude of the point
        lat (float): latitude of the point

    Returns:
        int: EPSG code
    """
    # UTM zone number starts from 1 at longitude -180,
    # and increments by 1 every 6 degrees of longitude
    zone = int((lon + 180) // 6 + 1)

    return utm_zone_to_epsg(zone, lat > 0)


def pyproj_transform(x, y, in_crs, out_crs, z=None):
    """
    Wrapper around pyproj to convert coordinates from an EPSG system to another.

    Args:
        x (scalar or array): x coordinate(s), expressed in in_crs
        y (scalar or array): y coordinate(s), expressed in in_crs
        in_crs (pyproj.crs.CRS or int): input coordinate reference system or EPSG code
        out_crs (pyproj.crs.CRS or int): output coordinate reference system or EPSG code
        z (scalar or array): z coordinate(s), expressed in in_crs

    Returns:
        scalar or array: x coordinate(s), expressed in out_crs
        scalar or array: y coordinate(s), expressed in out_crs
        scalar or array (optional if z): z coordinate(s), expressed in out_crs
    """
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True)
    if z is None:
        return transformer.transform(x, y)
    else:
        return transformer.transform(x, y, z)


def pyproj_lonlat_to_epsg(lon, lat, epsg):
    return pyproj_transform(lon, lat, 4326, epsg)


def pyproj_epsg_to_lonlat(x, y, epsg):
    return pyproj_transform(x, y, epsg, 4326)


def pyproj_lonlat_to_utm(lon, lat, epsg=None):
    if epsg is None:
        epsg = compute_epsg(np.mean(lon), np.mean(lat))
    x, y = pyproj_lonlat_to_epsg(lon, lat, epsg)
    return x, y, epsg


def utm_bounding_box_from_lonlat_aoi(aoi):
    """
    Computes the UTM bounding box (min_easting, min_northing, max_easting,
    max_northing)  of a projected AOI.

    Args:
        aoi (geojson.Polygon): GeoJSON polygon representing the AOI expressed in (long, lat)

    Return:
        min_easting, min_northing, max_easting, max_northing: the coordinates
        of the top-left corner and lower-right corners of the aoi in UTM coords
    """
    lons, lats = np.array(aoi["coordinates"][0]).T
    east, north, zone = lonlat_to_utm(lons, lats)
    pts = list(zip(east, north))
    emin, nmin, deltae, deltan = bounding_box2D(pts)
    return emin, emin + deltae, nmin, nmin + deltan


def simple_equalization_8bit(im, percentiles=5):
    """
    Simple 8-bit requantization by linear stretching.

    Args:
        im (np.array): image to requantize
        percentiles (int): percentage of the darkest and brightest pixels to saturate

    Returns:
        numpy array with the quantized uint8 image
    """
    import numpy as np

    mi, ma = np.percentile(im[np.isfinite(im)], (percentiles, 100 - percentiles))
    im = np.clip(im, mi, ma)
    im = (im - mi) / (ma - mi) * 255  # scale
    return im.astype(np.uint8)


def simplest_color_balance_8bit(im, percentiles=5):
    """
    Simple 8-bit requantization by linear stretching.

    Args:
        im (np.array): image to requantize
        percentiles (int): percentage of the darkest and brightest pixels to saturate

    Returns:
        numpy array with the quantized uint8 image
    """
    import numpy as np

    mi, ma = np.percentile(im[np.isfinite(im)], (percentiles, 100 - percentiles))
    im = np.clip(im, mi, ma)
    im = (im - mi) / (ma - mi) * 255  # scale
    return im.astype(np.uint8)


def matrix_translation(x, y):
    """
    Return the (3, 3) matrix representing a 2D shift in homogeneous coordinates.
    """
    t = np.eye(3)
    t[0, 2] = x
    t[1, 2] = y
    return t


def get_angle_from_cos_and_sin(c, s):
    """
    Computes x in ]-pi, pi] such that cos(x) = c and sin(x) = s.
    """
    if s >= 0:
        return np.arccos(c)
    else:
        return -np.arccos(c)


def find_key_in_geojson(it, q):
    """
    Traverses the geojson object it (dictionary mixed with lists)
    in a depth first order and returns the first ocurrence of the
    keywork q, otherwise returns None
    example:
        aoi = find_key_in_geojson(geojson.loads(geojsonstring),'geometry')
    """
    if isinstance(it, list):
        for item in it:
            t = find_key_in_geojson(item, q)
            if t is not None:
                return t
        return None
    elif isinstance(it, dict):
        if q in it.keys():
            return it[q]
        for key in it.keys():
            t = find_key_in_geojson(it[key], q)
            if t is not None:
                return t
        return None

    else:
        return None


def get_image_lonlat_aoi(rpc, h, w, z):
    """
    Returns the geojson polygon of the image footprint in lonlat coordinates
    """
    cols, rows, alts = [0, w, w, 0], [0, 0, h, h], [z] * 4
    lons, lats = rpc.localization(cols, rows, alts)
    lonlat_coords = np.vstack((lons, lats)).T
    geojson_polygon = {"coordinates": [lonlat_coords.tolist()], "type": "Polygon"}
    x_c = lons.min() + (lons.max() - lons.min()) / 2
    y_c = lats.min() + (lats.max() - lats.min()) / 2
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon


def generate_json_metadata(crop_fname, z):
    """
    Generates JSON metadata for a cropped GeoTIFF image.

    Args:
        crop_fname (str): Path to the cropped GeoTIFF file.

    Returns:
        dict: Metadata including image dimensions, sun elevation, sun azimuth,
              acquisition date, geojson footprint, and RPC model.
    """
    crop_img = readGTIFF(crop_fname).squeeze()
    crop_shape = crop_img.shape
    src = rasterio.open(crop_fname, "r")
    original_rpc = rpcm.RPCModel(src.tags(ns="RPC"))
    output_dict = {
        "img": crop_fname.split("/")[-1],  # TODO: check if naming is correct
        "height": crop_shape[0],
        "width": crop_shape[1],
        "sun_elevation": src.tags().get("NITF_USE00A_SUN_EL", None),
        "sun_azimuth": src.tags().get("NITF_USE00A_SUN_AZ", None),
        "acquisition_date": src.tags().get("NITF_STDIDC_ACQUISITION_DATE", None),
        "geojson": get_image_lonlat_aoi(original_rpc, crop_shape[0], crop_shape[1], z),
        "rpc": original_rpc.__dict__,
    }
    src.close()
    return output_dict


def create_footprint(list_of_tifs, method="union"):
    """
    Create a footprint using the specified method.

    Args:
        list_of_tifs (list): List of GeoTIFF file paths.
        method (str): Method to create the footprint. Options are 'union', 'intersection', 'pairwise_intersections'.

    Returns:
        shapely.geometry.Polygon: Resulting footprint polygon.
    """
    individual_footprints = []

    for f in list_of_tifs:
        footprint = lon_lat_image_footprint(f)
        polygon = shape(footprint)
        individual_footprints.append(polygon)

    if method == "union":
        return unary_union(individual_footprints)
    elif method == "intersection":
        intersection = individual_footprints[0]
        for footprint in individual_footprints[1:]:
            intersection = intersection.intersection(footprint)
        return intersection
    elif method == "pairwise_intersections":
        pairwise_intersection = [
            a.intersection(b) for a, b in combinations(individual_footprints, 2)
        ]
        return unary_union(pairwise_intersection)
    else:
        raise ValueError(
            "Invalid method. Choose from 'union', 'intersection', or 'pairwise_intersections'."
        )


def tile_in_image(tile, image, z):
    corners = [
        (lon, lat) for lon, lat in tile.geometry.__geo_interface__["coordinates"][0]
    ]
    lons = [lon for lon, _ in corners]
    lats = [lat for _, lat in corners]
    xs_img, ys_img = rpcm.projection(image, lons, lats, z)
    metadata = readGTIFFmeta(image)[0]
    # If all xs_imgs are in the image, the tile is in the image
    return all([0 <= x < metadata["width"] for x in xs_img]) and all(
        [0 <= y < metadata["height"] for y in ys_img]
    )


def crop_geotiff_lonlat_aoi(geotiff_path, output_path, tile, z, channels=None):
    with rasterio.open(geotiff_path, "r") as src:
        profile = src.profile
        tags = src.tags()
    crop, x, y = rpcm.utils.crop_aoi(geotiff_path, tile.geometry.__geo_interface__, z)
    rpc = rpcm.rpc_from_geotiff(geotiff_path)
    rpc.row_offset -= y
    rpc.col_offset -= x
    not_pan = len(crop.shape) > 2
    if not_pan:
        profile["height"] = crop.shape[1]
        profile["width"] = crop.shape[2]
        if channels is not None:
            # Channels are 1-indexed
            channels = [ch - 1 for ch in channels]
            crop = crop[channels]
        profile["count"] = len(channels) if channels is not None else crop.shape[0]
    else:
        profile["height"] = crop.shape[0]
        profile["width"] = crop.shape[1]
        profile["count"] = 1
    with rasterio.open(output_path, "w", **profile) as dst:
        if not_pan:
            dst.write(crop)
        else:
            dst.write(crop, 1)
        dst.update_tags(**tags)
        dst.update_tags(ns="RPC", **rpc.to_geotiff_dict())


def drop_z(geometry):
    if geometry.has_z:  # Check if the geometry has a Z dimension
        # Convert the geometry to a 2D polygon by keeping only X and Y coordinates
        return Polygon([(x, y) for x, y, z in geometry.exterior.coords])
    return geometry  # If it's already 2D, return as-is


def get_z_from_tile(tile, dem, dem_transform):
    """
    Get the median elevation value from a Digital Elevation Model (DEM) for a given tile.
    """
    mask = geometry_mask(
        [tile.geometry], out_shape=dem.shape, transform=dem_transform, invert=True
    )
    dem_values = dem[mask]
    return np.nanmedian(dem_values).item()


def pixel_to_geo(coords, dsm_transform):
    """Convert pixel coordinates to geographic coordinates using DSM metadata."""
    return np.array(rasterio.transform.xy(dsm_transform, coords[:, 0], coords[:, 1]))


def project_shadows(img_path, lon, lat, z):
    """Project shadows using RPC model."""
    return rpcm.projection(img_path, lon, lat, z)


def upsample_dsm(dsm_path, upscale_factor=4):
    """Upsample DSM using pixel-as-area approach."""
    src = rasterio.open(dsm_path)
    meta = src.meta.copy()

    # Calculate new dimensions
    original_height, original_width = src.shape
    new_height = int(original_height * upscale_factor)
    new_width = int(original_width * upscale_factor)

    # Calculate the pixel size in the new resolution
    pixel_size_x = src.transform[0] / upscale_factor
    pixel_size_y = abs(src.transform[4]) / upscale_factor

    # Explicitly recalculate the transform to ensure alignment with the original grid
    transform_upscaled = rasterio.transform.from_origin(
        src.transform.c,  # x coordinate of the upper-left corner
        src.transform.f,  # y coordinate of the upper-left corner
        pixel_size_x,
        pixel_size_y,  # new pixel size
    )

    # Update metadata
    meta.update(
        {"height": new_height, "width": new_width, "transform": transform_upscaled}
    )

    # Resample DSM to increase resolution
    dsm_upscaled = src.read(
        out_shape=(1, new_height, new_width), resampling=Resampling.bilinear
    )
    dsm_upscaled = dsm_upscaled.squeeze()

    # Create UTM coordinates for the upscaled grid, adjusted for pixel centers
    x = np.linspace(
        transform_upscaled[2] + pixel_size_x / 2,
        transform_upscaled[2] + pixel_size_x * new_width - pixel_size_x / 2,
        new_width,
    )
    y = np.linspace(
        transform_upscaled[5] - pixel_size_y / 2,
        transform_upscaled[5] - pixel_size_y * new_height + pixel_size_y / 2,
        new_height,
    )

    xy_utm_upscaled = np.array(np.meshgrid(x, y))

    src.close()

    return dsm_upscaled, xy_utm_upscaled, meta


def calculate_shadow_params(elevation, azimuth):
    #### p - - >
    #### q
    #### |
    #### |
    #### v
    # a slope

    elevation_rad = math.radians(float(elevation))
    azimuth_rad = math.radians(float(azimuth))

    p = -math.sin(azimuth_rad) * math.cos(elevation_rad)
    q = math.cos(azimuth_rad) * math.cos(elevation_rad)

    a = math.tan(elevation_rad)

    return p, q, a


# Function to load rpc from json
def rpc_from_json(json_path, return_dict=False):
    with open(json_path) as f:
        d = json.load(f)
    if return_dict:
        return d["rpc"]
    return rpcm.RPCModel(d["rpc"], dict_format="rpcm")


# Function to save geo tif
def save_geotiff(data, transform, crs, filename):
    # Create the file
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def get_latest_dsm_path(dataset_dir, aoi_id, city):
    """Get the latest DSM path for a given AOI and city."""
    # If easy to identify, we can manually select the DSM for each city
    # If not we'll sort the ones by name and select the first one
    LATEST_DSM = {
        "JAX": "FL_Peninsular_FDEM_Duval_2018",
        "OMA": "IA_FullState",
        "UCSD": "CA_SanDiegoQL2_2014",
    }

    possible_gt_dsm_files = [
        f for f in os.listdir(f"{dataset_dir}/dsm/{aoi_id}") if f.endswith("dsm_min.tif")
    ]
    if f"{aoi_id}_{LATEST_DSM.get(city)}_dsm_min.tif" in possible_gt_dsm_files:
        return f"{dataset_dir}/dsm/{aoi_id}/{aoi_id}_{LATEST_DSM.get(city)}_dsm_min.tif"
    else:
        latest_file = sorted(possible_gt_dsm_files, key=lambda x: x.split("_")[-2])[-1]
        return f"{dataset_dir}/dsm/{aoi_id}/{latest_file}"
