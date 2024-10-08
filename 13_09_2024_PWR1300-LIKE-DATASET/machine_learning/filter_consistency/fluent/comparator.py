import swigodessa as so
import abc
import typing
import logging
import fluent.path as flpa
import fluent.types as flty

class Diff:
	def __init__(self, path:flpa.Path) -> None:
		self.path = path

	def __str__(self) -> str:
		return F"Diff at path {self.path.__str__()}"


class NumberDiff(Diff):

	def __init__(self,path:flpa.Path,va,vb):
		super().__init__(path)
		self.va=va
		self.vb=vb
		self.diff = ( abs(va)+abs(vb) ) / 2.0
		if( self.diff!=0.0 ):
			self.diff = abs(va-vb) / self.diff
		self.diff = abs(self.diff)

	def __str__(self) -> str:
		return f"Number {self.va} != {self.vb} at path {self.path}"



class StrDiff(Diff):

	def __init__(self,path:flpa.Path,va:str,vb:str):
		super().__init__(path)
		self.va = va
		self.vb = vb

class AnyDiff(Diff):

	def __init__(self,path:flpa.FamilyPath,va:typing.Any,vb:typing.Any):
		super().__init__(path)
		self.va = va
		self.vb = vb

class PathDiff(Diff):
	def __init__(self, path: flpa.Path) -> None:
		super().__init__(path)

Diffs = typing.Iterable[Diff]

class Comparator(abc.ABC):

	def __init__(self, first:so.odbase, second:so.odbase) -> None:
		super().__init__()
		self.first = first
		self.second = second

	def compare(self, path:flpa.Path) -> Diffs:
		if path.check(self.first) and path.check(self.second):
			if path.typ == so.od_base:
				yield from self.compare_base(path)
			elif path.typ == so.od_r0:
				yield from self.compare_float(path)
			elif path.typ == so.od_r1:
				yield from self.compare_r1(path)
			elif path.typ == so.od_r2:
				yield from self.compare_r2(path)
			elif path.typ == so.od_r3:
				yield from self.compare_r3(path)
			elif path.typ == so.od_rg:
				yield from self.compare_rg(path)
			elif path.typ == so.od_i0:
				yield from self.compare_int(path)
			elif path.typ == so.od_i1:
				yield from self.compare_i1(path)
			elif path.typ == so.od_i2:
				yield from self.compare_i2(path)
			elif path.typ == so.od_ig:
				yield from self.compare_ig(path)
			elif path.typ == so.od_c0:
				yield from self.compare_str(path)
			elif path.typ == so.od_t:
				yield from self.compare_text(path)
			elif path.typ == so.od_c1:
				yield from self.compare_c1(path)
			else:
				yield from self.compare_any(path)

	def compare_base(self, path:flpa.Path[so.odbase]) -> Diffs:
		for new_path in flpa.enumerate_base(path, self.first):
			#logging.debug(f"Comparing {new_path}")
			yield from self.compare(new_path)

	def compare_str(self, path:flpa.Path[str]) -> Diffs:
		first_str = path[self.first]
		second_str = path[self.second]
		if first_str != second_str:
			yield StrDiff(path, first_str, second_str)
	
	def compare_text(self, path:flpa.Path[so.odt]) -> Diffs:
		first_str = so.odt_get(path[self.first])
		second_str = so.odt_get(path[self.second])
		if first_str != second_str:
			yield StrDiff(path, first_str, second_str)

	def compare_float(self, path:flpa.Path[float])-> Diffs:
		first_float = path[self.first]
		second_float = path[self.second]
		if first_float != second_float:
			yield NumberDiff(path, first_float, second_float)

	def compare_int(self, path:flpa.Path[int])-> Diffs:
		first_int = path[self.first]
		second_int = path[self.second]
		if first_int != second_int:
			yield NumberDiff(path, first_int, second_int)

	def compare_r1(self, path:flpa.Path[so.odr1])-> Diffs:
		for new_path in flpa.enumerate_r1(path, self.first):
			yield from self.compare(new_path)
	
	def compare_r2(self, path:flpa.Path[so.odr2])-> Diffs:
		for new_path in flpa.enumerate_r2(path, self.first):
			yield from self.compare(new_path)

	def compare_r3(self, path:flpa.Path[so.odr3])-> Diffs:
		for new_path in flpa.enumerate_r3(path, self.first):
			yield from self.compare(new_path)

	def compare_rg(self, path:flpa.Path[so.odrg])-> Diffs:
		for new_path in flpa.enumerate_rg(path, self.first):
			yield from self.compare(new_path)

	def compare_i1(self, path:flpa.Path[so.odi1])-> Diffs:
		for new_path in flpa.enumerate_i1(path, self.first):
			yield from self.compare(new_path)

	def compare_i2(self, path:flpa.Path[so.odi2])-> Diffs:
		for new_path in flpa.enumerate_i2(path, self.first):
			yield from self.compare(new_path)

	def compare_c1(self, path:flpa.Path[so.odc1])-> Diffs:
		for new_path in flpa.enumerate_c1(path, self.first):
			yield from self.compare(new_path)

	def compare_ig(self, path:flpa.Path[so.odig])-> Diffs:
		for new_path in flpa.enumerate_ig(path, self.first):
			yield from self.compare(new_path)

	def compare_any(self, path:flpa.Path) -> Diffs:
		logging.warning(f"Comparing any : currently ignoring (typ:{flty.typ_names[path.typ]})")
		return []