#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np


def remove_non_flow_metabolites(nu):
    is_metabolite_flow = (nu > 0).any(1) & (nu < 0).any(1)
    is_all_metabolite_flow = is_metabolite_flow.all()
    if not is_all_metabolite_flow:
        non_flow_metabolites = nu.loc[~is_metabolite_flow].index.tolist()
        print(f'{", ".join(non_flow_metabolites)} do not flow.')
        nu = nu.drop(non_flow_metabolites, axis=0)
    # deleting empty reactions
    nu = nu.loc[:, abs(nu).sum() != 0]
    return nu


def split_number_metabolite(string):
    number_str = ''
    metab_name = ''
    is_number = True
    for i, s in enumerate(string):
        if(s in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']) and is_number:
            number_str = number_str + s
        else:
            is_number = False
            metab_name = metab_name + s
    if number_str == '':
        number_str = '1'
    
    number = int(number_str) if number_str.find('.') == -1 else float(number_str)
    return [number, metab_name.lstrip()]


class ReactionFormulaParser():
    def __init__(self, data_path, ignored_reactions=['BIO'],
                 ignored_metabolites=['Export', 'H[c]', 'NH3[c]'],
                 ignored_metabolite_keywords=['[ex]', '[e]']):
        data = pd.read_csv(data_path)
        if not 'Reaction' in set(data.columns.tolist()):
            data['Reaction'] = ['Reaction{0:02d}'.format(index+1) 
                                for index in data.index.tolist()]
        if not ignored_reactions == None:
            for ignored_reaction in ignored_reactions:
                if ignored_reaction in set(data['Reaction'].tolist()):
                    data = data.loc[data['Reaction']!=ignored_reaction]
        self.data = data
        self.ignored_metabolites = ignored_metabolites
        self.ignored_metabolite_keywords = ignored_metabolite_keywords

    def extract_metabolite_names(self):
        metabolites = []
        for formula in self.data['Equation'].tolist():
            metabolites += formula.split(' ')
        metabolites = list(set(metabolites))
        metabolites = [split_number_metabolite(string)[1] for string in metabolites]
        metabolites = sorted(list(set(metabolites) - set(['', '+', '-->', '<=>'])))
        self.metabolites = pd.Series(metabolites)

    def split_reversible_reactions(self):
        self.reversible_reaction_names = []
        for i, reaction_name in enumerate(self.data['Reaction'].tolist()):
            formula = self.data['Equation'].iloc[i]
            if '<=>' in formula:
                self.reversible_reaction_names.append(reaction_name + '_r')

    def add_coef_to_nu(self, str_input_metab, str_output_metab, reaction_name):
        input_metabolites = str_input_metab.split(' + ')
        input_metabolites = [split_number_metabolite(string) 
                             for string in input_metabolites]
        output_metabolites = str_output_metab.split(' + ')
        output_metabolites = [split_number_metabolite(string) 
                              for string in output_metabolites]
        for number, input_metabolite in input_metabolites:
            self.nu.loc[input_metabolite, reaction_name] = -number
        for number, output_metabolite in output_metabolites:
            self.nu.loc[output_metabolite, reaction_name] = number

    def make_stoichiometric_matrix(self):
        reactions = sorted(self.data['Reaction'].tolist() + self.reversible_reaction_names)
        nu = pd.DataFrame(np.zeros((len(self.metabolites.index), len(reactions))), 
                          index=self.metabolites, 
                          columns=reactions, 
                          dtype=int)
        nu.index.name = 'Compound'
        self.nu = nu
        for i, reaction_name in enumerate(self.data['Reaction'].tolist()):
            formula = self.data['Equation'].iloc[i]
            str_input_metab = None
            str_output_metab = None
            if '<=>' in formula:
                str_input_metab, str_output_metab = formula.split(' <=> ')
                self.add_coef_to_nu(str_output_metab, str_input_metab, 
                                    reaction_name + '_r')
            if '-->' in formula:
                str_input_metab, str_output_metab = formula.split(' --> ')
            self.add_coef_to_nu(str_input_metab, str_output_metab, 
                                reaction_name)
        if not self.ignored_metabolites == None:
            for ignored_metabolite in self.ignored_metabolites:
                if ignored_metabolite in self.nu.index.tolist():
                    self.nu = self.nu.drop([ignored_metabolite])
        if not self.ignored_metabolite_keywords == None:
            for ignored_metabolite_keyword in self.ignored_metabolite_keywords:
                self.nu = self.nu[self.nu.index.str.find(ignored_metabolite_keyword) == -1]
        self.nu = self.nu.loc[:, abs(self.nu).sum() != 0]
        self.nu = self.nu.drop_duplicates()

    def parse(self):
        self.extract_metabolite_names()
        self.split_reversible_reactions()
        self.make_stoichiometric_matrix()