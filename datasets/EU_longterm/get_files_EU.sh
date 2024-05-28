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
    wget --show-progress \
      https://drive.utbm.fr/s/wneajmi6KiADqiB \
      https://drive.utbm.fr/s/6NcE2GSqNdGyELg \
      https://drive.utbm.fr/s/5ESADzCZ838At4b \
      https://drive.utbm.fr/s/JoB5gHwaEfDA8ga \
      https://drive.utbm.fr/s/GsbwwTkCDZFzNQe \
      https://drive.utbm.fr/s/WdFBbSk5c72TdTw \
      https://drive.utbm.fr/s/R99NdcaQWcejaH4 \
      https://drive.utbm.fr/s/6WfczpWcE8ce9t4 \
      https://drive.utbm.fr/s/FqyM5AFnfNnM4DA \
      https://drive.utbm.fr/s/WNf8ALLdtQokX3r \
      https://drive.utbm.fr/s/PEq5roAeLj3y9Rf \
      https://drive.utbm.fr/s/48pstJgSz9CniHG \
      https://drive.utbm.fr/s/fmBYFizza4n4L52 \
      https://drive.utbm.fr/s/PBn6SAWPPC73cco \
      https://drive.utbm.fr/s/SXD6FnZK3WFSwTS \
      https://drive.utbm.fr/s/tqYN75r5A3Cdzea \
      https://drive.utbm.fr/s/iGP3tBX2kxMy3DQ \
      https://drive.utbm.fr/s/PRLWKX3MLQJt5XD \
      https://drive.utbm.fr/s/x2aGgxC2jcXWTRN \
      https://drive.utbm.fr/s/Y2fnAfzgNGdS8Sj \
      https://drive.utbm.fr/s/YiX3DWfpmRKGKMX
    ;;

  "-roundabouts")
    wget --show-progress \
      https://drive.utbm.fr/s/xk6K4Rg8EGC6San \
      https://drive.utbm.fr/s/8Q6QcngjqDFqRcq \
      https://drive.utbm.fr/s/wn7RCYayNZJNxMj \
      https://drive.utbm.fr/s/d4jA8r2bbXG59kw
    ;;

  "-challenges")
    wget --show-progress \
      https://drive.utbm.fr/s/p3PinX5qQBxSdz9 \
      https://drive.utbm.fr/s/Nay2pTMpLgWX2tp \
      https://drive.utbm.fr/s/2b46iNkeJtdQ5BW \
      https://drive.utbm.fr/s/2mfBPXxKc4TJbRc \
      https://drive.utbm.fr/s/QpyJPbgaiG3S3dJ \
      https://drive.utbm.fr/s/JyNLiNMB9ZNiyg9 \
      https://drive.utbm.fr/s/9j7LwoA5T4FF6AC \
      https://drive.utbm.fr/s/LPjYJQxC7J7QBTN
    ;;

  "-all")
    wget --show-progress \
      https://drive.utbm.fr/s/wneajmi6KiADqiB \
      https://drive.utbm.fr/s/6NcE2GSqNdGyELg \
      https://drive.utbm.fr/s/5ESADzCZ838At4b \
      https://drive.utbm.fr/s/JoB5gHwaEfDA8ga \
      https://drive.utbm.fr/s/GsbwwTkCDZFzNQe \
      https://drive.utbm.fr/s/WdFBbSk5c72TdTw \
      https://drive.utbm.fr/s/R99NdcaQWcejaH4 \
      https://drive.utbm.fr/s/6WfczpWcE8ce9t4 \
      https://drive.utbm.fr/s/FqyM5AFnfNnM4DA \
      https://drive.utbm.fr/s/WNf8ALLdtQokX3r \
      https://drive.utbm.fr/s/PEq5roAeLj3y9Rf \
      https://drive.utbm.fr/s/48pstJgSz9CniHG \
      https://drive.utbm.fr/s/fmBYFizza4n4L52 \
      https://drive.utbm.fr/s/PBn6SAWPPC73cco \
      https://drive.utbm.fr/s/SXD6FnZK3WFSwTS \
      https://drive.utbm.fr/s/tqYN75r5A3Cdzea \
      https://drive.utbm.fr/s/iGP3tBX2kxMy3DQ \
      https://drive.utbm.fr/s/PRLWKX3MLQJt5XD \
      https://drive.utbm.fr/s/x2aGgxC2jcXWTRN \
      https://drive.utbm.fr/s/Y2fnAfzgNGdS8Sj \
      https://drive.utbm.fr/s/YiX3DWfpmRKGKMX \
      https://drive.utbm.fr/s/xk6K4Rg8EGC6San \
      https://drive.utbm.fr/s/8Q6QcngjqDFqRcq \
      https://drive.utbm.fr/s/wn7RCYayNZJNxMj \
      https://drive.utbm.fr/s/d4jA8r2bbXG59kw \
      https://drive.utbm.fr/s/p3PinX5qQBxSdz9 \
      https://drive.utbm.fr/s/Nay2pTMpLgWX2tp \
      https://drive.utbm.fr/s/2b46iNkeJtdQ5BW \
      https://drive.utbm.fr/s/2mfBPXxKc4TJbRc \
      https://drive.utbm.fr/s/QpyJPbgaiG3S3dJ \
      https://drive.utbm.fr/s/JyNLiNMB9ZNiyg9 \
      https://drive.utbm.fr/s/9j7LwoA5T4FF6AC \
      https://drive.utbm.fr/s/LPjYJQxC7J7QBTN

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