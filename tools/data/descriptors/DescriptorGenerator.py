from __future__ import print_function
from rdkit import Chem
import logging, numpy, sys
import pandas as pd
import pandas_flavor as pf
import sys
import numpy as np
import sys
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping

MAX_CACHE = 0
ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)


def getsize(obj_0):
    _seen_ids = set()

    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, ZERO_DEPTH_BASES):
            pass
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'):
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size

    return inner(obj_0)


class DescriptorGenerator:
    REGISTRY = {}
    NAME = None

    def __init__(self):
        try:
            self.REGISTRY[self.NAME.lower()] = self
        except:
            logging.exception("DescriptorGenerator must have a NAME (self.NAME)")
            raise
        self.columns = []
        self.cache = {}
        self.cache_hit = 0
        self.cache_miss = 0

    def molFromSmiles(self, smiles):
        return Chem.MolFromSmiles(smiles)

    def molFromMol(self, mol):
        return mol

    def GetColumns(self):
        if self.NAME:
            return [(self.NAME + "_calculated", numpy.bool)] + self.columns
        return self.columns

    def calculateMol(self, m, smiles, internalParsing):
        raise NotImplementedError

    def processMol(self, m, smiles, internalParsing=False):
        if not internalParsing:
            m = self.molFromMol(m)
        res = self.calculateMol(m, smiles, internalParsing)
        if None in res:
            logging.error("None in res")
            columns = self.GetColumns()
            for idx, v in enumerate(res):
                if v is None:
                    if self.NAME:
                        logging.error("At least one result: %s(%s) failed: %s",
                                      self.NAME,
                                      columns[idx + 1][0],
                                      smiles)
                        res[idx] = columns[idx + 1][1]()
                    else:
                        logging.error("At least one result: %s failed: %s",
                                      columns[idx][0],
                                      smiles)
                        res[idx] = columns[idx][1]()
            logging.info("res %r", res)
            if type(res) == list:
                res.insert(0, False)
            else:
                np.insert(res, 0, 3)
        else:
            if type(res) == list:
                res.insert(0, True)
            else:
                np.insert(res, 0, -1)
        return res

    def processMols(self, mols, smiles, internalParsing=False):
        if len(mols) != len(smiles):
            raise ValueError("Number of molecules does not match number of unparsed molecules")
        result = [self.processMol(m, smile, internalParsing)
                  for m, smile in zip(mols, smiles)]
        assert len(result) == len(mols)
        return result

    def process(self, smiles):
        try:
            mol = self.molFromSmiles(smiles)
        except:
            return None
        if mol == None:
            return None
        return self.processMol(mol, smiles, internalParsing=True)

    def processSmiles(self, smiles, keep_mols=True):
        mols = []
        allmols = []
        indices = []
        goodsmiles = []
        _results = []
        if MAX_CACHE:
            for i, smile in enumerate(smiles):
                res, m = self.cache.get(smile, (None, None))
                if res:
                    _results.append((i, res))
                    if keep_mols:
                        allmols.append(m)
                else:
                    m = self.molFromSmiles(smile)
                    if m:
                        mols.append(m)
                        indices.append(i)
                        goodsmiles.append(smile)
                    if keep_mols:
                        allmols.append(m)
        else:
            for i, smile in enumerate(smiles):
                m = self.molFromSmiles(smile)
                if m:
                    mols.append(m)
                    indices.append(i)
                    goodsmiles.append(smile)
                if keep_mols:
                    allmols.append(m)

        if len(smiles) + len(self.cache) > MAX_CACHE:
            self.cache.clear()
        if len(_results) == len(smiles):
            all_results = [r[1] for r in _results]
            return allmols, all_results
        elif len(_results) == 0:
            results = self.processMols(mols, goodsmiles, internalParsing=True)
            if MAX_CACHE:
                if len(indices) == len(smiles):
                    for smile, res, m in zip(smiles, results, allmols):
                        self.cache[smile] = res, m
                return mols, results
            all_results = [None] * len(smiles)
            for idx, result, m in zip(indices, results, allmols):
                self.cache[smiles[idx]] = result, m
                all_results[idx] = result
            return allmols, all_results
        else:
            results = self.processMols(mols, goodsmiles, internalParsing=True)
            all_results = [None] * len(smiles)
            for i, res in _results:
                all_results[i] = res
            for idx, result, m in zip(indices, results, allmols):
                if MAX_CACHE:
                    self.cache[smiles[idx]] = result, m
                all_results[idx] = result
            return allmols, all_results

    def processCtab(self, ctab):
        raise NotImplementedError

    def processSDF(self, sdf):
        raise NotImplementedError


class Container(DescriptorGenerator):
    def __init__(self, generators):
        self.generators = generators
        columns = self.columns = []
        for g in generators:
            columns.extend(g.GetColumns())
        self.cache = {}

    def processMol(self, m, smiles, internalParsing=False):
        results = []
        for g in self.generators:
            results.extend(g.processMol(m, smiles, internalParsing))
        return results

    def processMols(self, mols, smiles, internalParsing=False):
        results = []
        for m in mols:
            results.append([])
        for g in self.generators:
            for result, newresults in zip(results, g.processMols(mols, smiles, internalParsing)):
                result.extend(newresults)
        return results


def MakeGenerator(generator_names):
    if not len(generator_names):
        logging.warning("MakeGenerator called with no generator names")
        raise ValueError("MakeGenerator called with no generator names")
    generators = []
    for name in generator_names:
        try:
            d = DescriptorGenerator.REGISTRY[name.lower()]
            generators.append(d)
        except:
            logging.exception("No DescriptorGenerator found named %s\nCurrently registered descriptors:\n\t%s", name,
                              "\n\t".join(sorted(DescriptorGenerator.REGISTRY.keys())))
            raise
    if len(generators) > 1:
        return Container(generators)
    if len(generators):
        return generators[0]


@pf.register_dataframe_method
def create_descriptors(df: pd.DataFrame, mols_column_name: str, generator_names: list):
    generator = MakeGenerator(generator_names)
    mols = df[mols_column_name]
    if len(mols):
        if type(mols[0]) == str:
            _, results = generator.processSmiles(mols)
        else:
            results = generator.processMols(mols, [Chem.MolToSmiles(m) for m in mols])
    else:
        results = []
    fpdf = pd.DataFrame(results, columns=generator.GetColumns())
    fpdf.index = df.index
    return fpdf
