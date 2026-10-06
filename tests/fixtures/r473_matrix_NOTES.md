# Round 473 — JPEG reconstruction matrix fixtures (`r473_*`)

Locally generated, 2026-10-07. One 100×60 source raster (ImageMagick
`plasma:fractal -seed 7`, written as binary PPM), encoded by `cjpeg`
into every cell of

    sampling {4:4:4, 4:2:0, 4:2:2, 4:4:0} × {baseline, progressive} × {no DRI, DRI}

and losslessly recompressed with `cjxl v0.12.0`. 100×60 is not a
multiple of any subsampled MCU (16×16 / 16×8 / 8×16), so every
subsampled class exercises the §F.2 MCU-padded block grids.

## Command lines

    magick -size 100x60 plasma:fractal -seed 7 base.png
    magick base.png -depth 8 base.ppm
    # per cell: S ∈ {1x1, 2x2, 2x1, 1x2}  (4:4:4, 4:2:0, 4:2:2, 4:4:0)
    cjpeg -quality 80 -sample S                       -outfile r473_<s>_base_nodri.jpg base.ppm
    cjpeg -quality 80 -sample S -restart 1            -outfile r473_<s>_base_dri.jpg   base.ppm
    cjpeg -quality 80 -sample S -progressive          -outfile r473_<s>_prog_nodri.jpg base.ppm
    cjpeg -quality 80 -sample S -progressive -restart 1 -outfile r473_<s>_prog_dri.jpg base.ppm
    cjxl --lossless_jpeg=1 r473_<cell>.jpg r473_<cell>.jxl

`cjxl` refused two cells with `EncodeImageJXL() failed`: `420_prog_dri`
and `422_prog_dri` (progressive + restart interval on a horizontally
subsampled source). Those two have no `.jxl` and are not pinned; the
4:4:4 and 4:4:0 progressive + DRI cells transcode fine.

4:1:1 (`-sample 4x1`) and 4:1:0 (`4x2`) are not representable:
`jpeg_upsampling` (ISO/IEC 18181-1 F.2) only describes {1,1}, {2,2},
{2,1} and {1,2} factors, and `cjxl` refuses them with the same error.

## `djpeg` references (`r473_<s>_djpeg.png`)

    djpeg -outfile ref.ppm r473_<s>_base_nodri.jpg
    magick ref.ppm -strip r473_<s>_djpeg.png

The baseline / progressive / DRI variants of one sampling class carry
identical coefficients, so one reference per class serves all of them
(`tests/r473_jpeg_pixels.rs` also pins that the variants decode to
byte-identical planes). The `*_djpeg.png` files next to the older
`r448_*` / `r451_*` / `jpeg_transcode` fixtures were produced the same
way from their `.jpg` siblings (`jpeg_transcode_original.jpg` for
`jpeg_transcode.jxl`).

## SHA-256

    b1687ab0f93934bf27db60d0b469311bd732e1908d8dc62bdd8b9264d0c59a78 r473_420_base_dri.jpg
    dddb8207290c12e82e1f45db55017ef09af1dad26d8671743b746dcfaa347fdf r473_420_base_nodri.jpg
    c90f55e18f53789170c1617f950fa2fca1daf8956c13f14ec3346aeca0765e15 r473_420_prog_nodri.jpg
    c03638b55d4dfde41fda3119dc4f4a7798f1047871b80e03ab5764c8a6dca032 r473_422_base_dri.jpg
    f63cab80a5cc2805dd51645131086a9cd4608c403cfe55a36d47cb9422d795bc r473_422_base_nodri.jpg
    b93553ebc2bbafb713ccdfea87e3cee852a216770e5e42412a4dbba076bd718a r473_422_prog_nodri.jpg
    b7d13f45f258b49469e4bec8242311ae27ead7fb73db77f3b52ee0798c1a1811 r473_440_base_dri.jpg
    a37dde9daeedf8b68a0dc1a08bd93262c1fb3a580d5bdeb483e7883ce474c970 r473_440_base_nodri.jpg
    dbd75bc12652b751b5f64786f61f045723d83b1b6a5e16f8f4f2d41109c4f4c4 r473_440_prog_dri.jpg
    5089d11a3807892b07424d1f615f2abf68bc1fca3d0c8033283dc75b53cdd104 r473_440_prog_nodri.jpg
    d6b32b5cd24e425586482f924e09791dcca6a4a5b9e7377a9b68dc3971c0b7e1 r473_444_base_dri.jpg
    fec62cc0b2a1eb307823dab18e3aecb23c1a46a31bab1b2fee4ab07e36969c5e r473_444_base_nodri.jpg
    5b50e72495fdadbc064e55b4599d3f3f8ad39ae5d661e700a00776bb67226a5d r473_444_prog_dri.jpg
    ac099dae59059d7db5212de392a79ad55669f17b4cc9f62f045f1adbaed48de1 r473_444_prog_nodri.jpg
    bfe37a9cd3418fe5011b652dc43eb079f5c712700a654efa0075a29c3b7b2b44 r473_420_base_dri.jxl
    8276debf26b92223b77476020d8586e1ac881b605c152b4bbb46502ae64ae048 r473_420_base_nodri.jxl
    d89838c69782af5b78ae32b5ce06aa264b66965f51a7cda313a4b873eb727751 r473_420_prog_nodri.jxl
    5d6e40e97926083ccfd7a8aec7638fc11b83450497387e76fbbddf7533a973e5 r473_422_base_dri.jxl
    bf11dbf3107820df8132117be14599cb47f361755828cddeb98b8cf5f07a4a9e r473_422_base_nodri.jxl
    732bf7667ee63c2d048ca28a1068100faca97768f71b357df4bb761f564d0c27 r473_422_prog_nodri.jxl
    f0ac518a71c93b0faec7bfcb174b3464f74bd0b94c7ecd7d461a7996c5c00c57 r473_440_base_dri.jxl
    7d640d39e8700179cee9d200858007ae609f94907b909a9fe69d395c6cc135bd r473_440_base_nodri.jxl
    7d297597a16e4323d2baab765a9780a2d6210b4b8014499a972b52d722889e4a r473_440_prog_dri.jxl
    207c112f4122c54c871e0054502a26ddfbaae4afc6d9e17b32e750d6275d9eab r473_440_prog_nodri.jxl
    2c929f6a4b18bbaf598af9c69e4f8c538c6eb2fd868c8ed797a33af0f533a0b2 r473_444_base_dri.jxl
    e4bd15a6697271b9d79259de9c861c09a5690cf10fcc9aae10cd2e3afef1cff1 r473_444_base_nodri.jxl
    1867b479195ae8c70780b9c9426e084b0ac3e634e295650da20ffa05dc57c454 r473_444_prog_dri.jxl
    56f1953da5a2c88d7daed3acefc920b7e9a222f47a0dd1670969eb1843df0472 r473_444_prog_nodri.jxl
    e13865c4c0f0ef4919df8f6ad328f1b1a849396e3623e2f97cb2530aea3c3ce5 r473_420_djpeg.png
    0a03a07ed502ab4c868fee1df3d2f6562285b600a442c5665fbccd3fbbeabec7 r473_422_djpeg.png
    f7d5bda61e4294019dc62a8ce76067b669521fda7c51dcc7c40b34514ce154e0 r473_440_djpeg.png
    b35f0ba3a3a17e44d973ac6740e8cb0cb4de2913ebe354198c8c63058e64da7c r473_444_djpeg.png
