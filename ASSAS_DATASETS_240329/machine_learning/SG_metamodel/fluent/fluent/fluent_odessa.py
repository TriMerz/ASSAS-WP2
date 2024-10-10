import typing
import pyastec as pyas

C_1D = typing.Tuple[int]
C_2D = typing.Tuple[int, int]
C_3D = typing.Tuple[int, int, int]
T = typing.TypeVar('T', pyas.odbase, float, pyas.odr1, pyas.odr2, pyas.odr3, int, pyas.odi1, pyas.odi2, str, pyas.odc1, pyas.odrg, pyas.odig, pyas.odelem, pyas.odt)
GT = typing.TypeVar('GT', pyas.odrg, pyas.odig)
AT = typing.TypeVar('AT', pyas.odc1, pyas.odi1, pyas.odi2, pyas.odr1, pyas.odr2, pyas.odr3)
X = typing.TypeVar('X', float, int, str)
C = typing.TypeVar('C', C_1D, C_2D, C_3D)

typ_names = {
    pyas.od_base:"base",
    pyas.od_r0:"float",
    pyas.od_r1:"r1",
    pyas.od_r2:"r2",
    pyas.od_r3:"r3",
    pyas.od_i0:"int",
    pyas.od_i1:"i1",
    pyas.od_i2:"i2",
    pyas.od_c0:"c0",
    pyas.od_c1:"c1",
    pyas.od_rg:"rg",
    pyas.od_ig:"ig",
    pyas.od_t:"text",
}