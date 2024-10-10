import abc
import typing
import logging
import pyastec as pyas
import fluent.path as pa
import fluent.enumerate as en

class Diff:
	def __init__(self, path:pa.Path) -> None:
		self.path = path

	def __str__(self) -> str:
		return F"Diff at path {self.path.__str__()}"

class NumberDiff(Diff):

	def __init__(self, path:pa.Path, first, second):
		super().__init__(path)
		self.first = first
		self.second = second

	def __str__(self) -> str:
		return f"Number {self.first} != {self.second} at path {self.path}"

class StrDiff(Diff):

	def __init__(self,path:pa.Path,va:str,vb:str):
		super().__init__(path)
		self.va = va
		self.vb = vb

class AnyDiff(Diff):

	def __init__(self,path:pa.FamilyPath,va:typing.Any,vb:typing.Any):
		super().__init__(path)
		self.va = va
		self.vb = vb

class PathDiff(Diff):
	def __init__(self, path: pa.Path, first:bool, second:bool) -> None:
		super().__init__(path)
		self.first = first
		self.second = second
	
	def __str__(self) -> str:
		return f"Path doesn't exist in {'first base' if self.first else ''} {'second base' if self.second else ''}"

Diffs = typing.Iterable[Diff]

class Comparator(abc.ABC):

	def __init__(self) -> None:
		super().__init__()
		self.family_type_compare_fun = {
			pyas.od_base:self.compare_base,
			pyas.od_t:self.compare_text,
			pyas.od_c0:self.compare_str,
			pyas.od_c1:self.compare_c1,
			pyas.od_i0:self.compare_int,
			pyas.od_i1:self.compare_i1,
			pyas.od_i2:self.compare_i2,
			pyas.od_ig:self.compare_ig,
			pyas.od_r0:self.compare_float,
			pyas.od_r1:self.compare_r1,
			pyas.od_r2:self.compare_r2,
			pyas.od_r3:self.compare_r3,
			pyas.od_rg:self.compare_rg,
		}
		self.array_type_compare_fun = {
			pyas.od_c1:self.compare_str,
			pyas.od_i1:self.compare_int,
			pyas.od_i2:self.compare_int,
			pyas.od_r1:self.compare_float,
			pyas.od_r2:self.compare_float,
			pyas.od_r3:self.compare_float,
		}
		self.group_type_compare_fun = {
			pyas.od_rg:self.compare_float,
			pyas.od_ig:self.compare_int,
		}

	def compare(self, paths:typing.Iterable[pa.Path], first:pyas.odbase, second:pyas.odbase) -> Diffs:
		for path in paths:
			check_first = path.exists_from(first)
			check_second = path.exists_from(second)
			if not check_first or not check_second:
				yield PathDiff(path, check_first, check_second)
			else:
				value_1 = path.get_from(first)
				value_2 = path.get_from(second)

				if isinstance(path, pa.Root):
					return []
				elif isinstance(path, pa.FamilyPath):
					if path.typ in self.family_type_compare_fun:
						yield from self.family_type_compare_fun[path.typ](path, value_1, value_2)
				elif isinstance(path, pa.ArrayElementPath):
					if path.parent.typ in self.array_type_compare_fun:
						yield from self.array_type_compare_fun[path.parent.typ](path, value_1, value_2)
				elif isinstance(path, pa.GroupElementPath):
					if path.parent.typ in self.group_type_compare_fun:
						yield from self.group_type_compare_fun[path.parent.typ](path, value_1, value_2)
				else:
					logging.error(f"Unhandled path type : {path}")

	def compare_base(self, path:pa.BasePath, first:pyas.odbase, second:pyas.odbase) -> Diffs:
		return []

	def compare_str(self, path:pa.StrFamilyPath, first:str, second:str) -> Diffs:
		if first != second:
			return [StrDiff(path, first, second)]
		else:
			return []
	
	def compare_text(self, path:pa.TextFamilyPath, first:pyas.odt, second:pyas.odt) -> Diffs:
		first_str = pyas.odt_get(first)
		second_str = pyas.odt_get(second)
		if first_str != second_str:
			return [StrDiff(path, first_str, second_str)]
		else:
			return []

	def compare_float(self, path:pa.FloatFamilyPath, first:float, second:float)-> Diffs:
		if first != second:
			return [NumberDiff(path, first, second)]
		else:
			return []

	def compare_int(self, path:pa.IntFamilyPath, first:int, second:int)-> Diffs:
		if first != second:
			return [NumberDiff(path, first, second)]
		else:
			return []

	def compare_r1(self, path:pa.R1FamilyPath, first:pyas.odr1, second:pyas.odr1)-> Diffs:
		return []
	
	def compare_r2(self, path:pa.R2FamilyPath, first:pyas.odr2, second:pyas.odr2)-> Diffs:
		return []

	def compare_r3(self, path:pa.R3FamilyPath, first:pyas.odr3, second:pyas.odr3)-> Diffs:
		return []

	def compare_rg(self, path:pa.RGFamilyPath, first:pyas.odrg, second:pyas.odrg)-> Diffs:
		return []

	def compare_i1(self, path:pa.I1FamilyPath, first:pyas.odi1, second:pyas.odi1)-> Diffs:
		return []

	def compare_i2(self,path:pa.I2FamilyPath,  first:pyas.odi2, second:pyas.odi2)-> Diffs:
		return []

	def compare_c1(self,path:pa.C1FamilyPath,  first:pyas.odc1, second:pyas.odc1)-> Diffs:
		return []

	def compare_ig(self,path:pa.IGFamilyPath,  first:pyas.odig, second:pyas.odig)-> Diffs:
		return []