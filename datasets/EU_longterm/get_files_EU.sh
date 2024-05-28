#!/bin/bash
# -----------------------------------------------------------------#
# script, which downloads the multi-sensor autonomous car data     #
# from the EU Long-term Dataset                                    #
# (https://epan-utbm.github.io/utbm_robocar_dataset/)              #
# -----------------------------------------------------------------#

echo "downloading the EU Long-term Dataset for object tracking ..."

case $1 in

  *"help"*)
    echo """
    Usage:
      --help        prints help
      -challenges   downloads only the challenge files
      -longterm     downloads the long-term data
      -roundabouts  downloads the data from roundabouts
      -all          downloads everything
    """
    ;;

  "-longterm")
    # non-image files:
    wget --show-progress \
      https://drive.utbm.fr/s/YiX3DWfpmRKGKMX/download/utbm_robocar_dataset_20180502_evening_xb3 \
      https://drive.utbm.fr/s/xk6K4Rg8EGC6San/download/utbm_robocar_dataset_20180502_noimage1.bag \
      https://drive.utbm.fr/s/Y2fnAfzgNGdS8Sj/download/utbm_robocar_dataset_20180502_noimage2.bag \
      https://drive.utbm.fr/s/PRLWKX3MLQJt5XD/download/utbm_robocar_dataset_20180713_noimage.bag \
      https://drive.utbm.fr/s/tqYN75r5A3Cdzea/download/utbm_robocar_dataset_20180716_noimage.bag \
      https://drive.utbm.fr/s/PBn6SAWPPC73cco/download/utbm_robocar_dataset_20180717_noimage.bag \
      https://drive.utbm.fr/s/48pstJgSz9CniHG/download/utbm_robocar_dataset_20180718_noimage.bag \
      https://drive.utbm.fr/s/WNf8ALLdtQokX3r/download/utbm_robocar_dataset_20180719_noimage.bag \
      https://drive.utbm.fr/s/6WfczpWcE8ce9t4/download/utbm_robocar_dataset_20180720_noimage.bag \
      https://drive.utbm.fr/s/WdFBbSk5c72TdTw/download/utbm_robocar_dataset_20190110_noimage.bag \
      https://drive.utbm.fr/s/JoB5gHwaEfDA8ga/download/utbm_robocar_dataset_20190131_noimage.bag \
      https://drive.utbm.fr/s/6NcE2GSqNdGyELg/download/utbm_robocar_dataset_20190418_noimage.bag;
    # image files have to be dealt with separately because each download is for multiple files:
    wget --show-progress -O utbm_robocar_dataset_20180502_image2.zip https://drive.utbm.fr/s/x2aGgxC2jcXWTRN/download
    wget --show-progress -O utbm_robocar_dataset_20180713_image.zip https://drive.utbm.fr/s/iGP3tBX2kxMy3DQ/download
    wget --show-progress -O utbm_robocar_dataset_20180716_image.zip https://drive.utbm.fr/s/SXD6FnZK3WFSwTS/download
    wget --show-progress -O utbm_robocar_dataset_20180717_image.zip https://drive.utbm.fr/s/fmBYFizza4n4L52/download
    wget --show-progress -O utbm_robocar_dataset_20180718_image.zip https://drive.utbm.fr/s/PEq5roAeLj3y9Rf/download
    wget --show-progress -O utbm_robocar_dataset_20180719_image.zip https://drive.utbm.fr/s/FqyM5AFnfNnM4DA/download
    wget --show-progress -O utbm_robocar_dataset_20180720_image.zip https://drive.utbm.fr/s/R99NdcaQWcejaH4/download
    wget --show-progress -O utbm_robocar_dataset_20190110_image.zip https://drive.utbm.fr/s/GsbwwTkCDZFzNQe/download
    wget --show-progress -O utbm_robocar_dataset_20190131_image.zip https://drive.utbm.fr/s/5ESADzCZ838At4b/download
    wget --show-progress -O utbm_robocar_dataset_20190418_image.zip https://drive.utbm.fr/s/wneajmi6KiADqiB/download
    ;;

  "-roundabouts")
    # non-image files:
    wget --show-progress \
      https://drive.utbm.fr/s/d4jA8r2bbXG59kw/download/utbm_robocar_dataset_20190412_roundabout_noimage.bag \
      https://drive.utbm.fr/s/wn7RCYayNZJNxMj/download/utbm_robocar_dataset_20190418_roundabout_noimage.bag;
    # image files:
    wget --show-progress -O utbm_robocar_dataset_20190412_roundabout_image.zip https://drive.utbm.fr/s/9yTDg7QbLpr6BLR/download
    wget --show-progress -O utbm_robocar_dataset_20190418_roundabout_image.zip https://drive.utbm.fr/s/8Q6QcngjqDFqRcq/download
    ;;

  "-challenges")
    wget --show-progress \
      https://drive.utbm.fr/s/p3PinX5qQBxSdz9/download/shared_zone.zip \
      https://drive.utbm.fr/s/Nay2pTMpLgWX2tp/download/construction_bypass.zip \
      https://drive.utbm.fr/s/2b46iNkeJtdQ5BW/download/roundabout.zip \
      https://drive.utbm.fr/s/2mfBPXxKc4TJbRc/download/snow.zip \
      https://drive.utbm.fr/s/QpyJPbgaiG3S3dJ/download/right_overtaking.zip \
      https://drive.utbm.fr/s/JyNLiNMB9ZNiyg9/download/crossing.zip \
      https://drive.utbm.fr/s/9j7LwoA5T4FF6AC/download/pigeon.zip \
      https://drive.utbm.fr/s/LPjYJQxC7J7QBTN/download/police.zip
    ;;

  "-all")
    wget --show-progress http://datasets.chronorobotics.tk/s/vVGheghiMYIF988/download \
         -O utbm_robotcar_dataset.zip

    ;;

  *)
    echo >&2 "Invalid option: $@";
    echo """
    Usage:
      --help        prints help
      -challenges   downloads only the challenge files
      -longterm     downloads the long-term data
      -roundabouts  downloads the data from roundabouts
      -all          downloads everything
    """;
    exit 1;
    ;;
    
esac




echo "done!"