"""
LEX-YACC parser class
Input: configurations (excel), dataframe to process
Step1: Dataframe check
Step2: YACC parse the configuration
Step3: Excute function
"""

import re
import logging
import sys

import pandas as pd
import numpy as np
from ply import lex, yacc


class PLYParser:

  # Constructor
  def __init__(self, configs, df):

    self.configs = configs
    self.df = df
    self.df_res = pd.DataFrame()
    # Python functions, mapped to token function
    self.python_funcs = {'ABS': self.df_abs,
                         'MIN': self.df_min,
                         'SUM': self.df_sum,
                         'MAX': self.df_max,
                         'MEDIAN': self.df_median,
                         'COUNT': self.df_count,
                         'AVERAGE': self.df_average,
                         'IF': self.df_if,
                         'LEN': self.df_len,
                         'SIN': self.df_sin,
                         'LOG10': self.df_log10,
                         'LOG': self.df_log,
                         'STDEV.P': self.df_stdevp,
                         'STDEV.S': self.df_stdevs}
    self.error_list=[]
    # Build the lex, yacc
    lex.lex(module=self)
    yacc.yacc(module=self, debug=False, write_tables=False)

  def assign(self, varname):
    df_s = self.df.get(varname, None)
    if df_s is None:
      return logging.error('The column {%s} is not found', varname)
    return df_s

  # Python functions
  @staticmethod
  def df_abs(sub_df):
    return abs(sub_df)  # return sub_df.abs()

  @staticmethod
  def df_min(sub_df, bycol=False):
    """Get the min"""
    if bycol:
      return pd.Series(np.repeat(sub_df.min(), sub_df.shape[0]))
    return sub_df.min(axis=1)

  @staticmethod
  def df_max(sub_df, bycol=False):
    """Get the max"""
    if bycol:
      return pd.Series(np.repeat(sub_df.max(), sub_df.shape[0]))
    return sub_df.max(axis=1)

  @staticmethod
  def df_sum(sub_df, bycol=False):
    """Get the sum"""
    if bycol:
      return pd.Series(np.repeat(sub_df.sum(), sub_df.shape[0]))
    return sub_df.sum(axis=1)

  @staticmethod
  def df_median(sub_df, bycol=False):
    """Get the median"""
    if bycol:
      return pd.Series(np.repeat(sub_df.median(), sub_df.shape[0]))

  @staticmethod
  def df_count(sub_df, bycol=False):
    """Count the number"""
    if bycol:
      return pd.Series(np.repeat(sub_df.count(), sub_df.shape[0]))
    return sub_df.count(axis=1)

  @staticmethod
  def df_average(sub_df, bycol=False):
    """Get the average"""
    if bycol:
      return pd.Series(np.repeat(sub_df.mean(), sub_df.shape[0]))
    return sub_df.mean(axis=1)

  def df_if(self, condition):
    """Excute if statement"""
    try:
      if_result = pd.Series(np.where(condition['condition'], condition['value1'],
                    condition['value2']))
    except:
      self.error_list.append(
        'There are problems with IF statements, the error message is:',
        sys.exc_info()[0])
      return None
    else:
      return if_result

  @staticmethod
  def df_sin(sub_df):
    return np.sin(sub_df)

  @staticmethod
  def df_log10(sub_df):
    """Get the log10 result"""
    return np.log10(sub_df)

  @staticmethod
  def df_log(sub_df):
    """Get the log result"""
    return np.log(sub_df)

  @staticmethod
  def df_len(sub_df):
    """Get the length"""
    return sub_df.size

  @staticmethod
  def df_stdevp(sub_df):
    """Get the population standard deviation"""
    return np.std(sub_df)

  @staticmethod
  def df_stdevs(sub_df):
    """Get the sample standard deviation"""
    return sub_df.std()

  def calc(self):
    for config in self.configs:
      # remove space
      config_ns = self.configs[config].strip()
      # print("{}:{}".format(config, config_ns))
      result = yacc.parse(config_ns)
      # Assign to the new col
      if isinstance(result, (pd.Series, float)):
        self.df_res[config] = result
    return self.df_res, self.error_list


class Excel2Pd(PLYParser):
  excel_funs = ['ABS', 'ACCRINT', 'ACOS', 'ACOSH', 'ACOT', 'ACOTH', 'ADD',
                'AGGREGATE', 'AND', 'ARABIC', 'ARGS2ARRAY', 'ASIN', 'ASINH',
                'ATAN', 'ATAN2', 'ATANH', 'AVEDEV', 'AVERAGE', 'AVERAGEA',
                'AVERAGEIF', 'AVERAGEIFS', 'BASE', 'BESSELI', 'BESSELJ',
                'BESSELK', 'BESSELY', 'BETA.DIST', 'BETA.INV', 'BETADIST',
                'BETAINV', 'BIN2DEC', 'BIN2HEX', 'BIN2OCT', 'BINOM.DIST',
                'BINOM.DIST.RANGE', 'BINOM.INV', 'BINOMDIST', 'BITAND',
                'BITLSHIFT', 'BITOR', 'BITRSHIFT', 'BITXOR', 'CEILING',
                'CEILINGMATH', 'CEILINGPRECISE', 'CHAR', 'CHISQ.DIST',
                'CHISQ.DIST.RT', 'CHISQ.INV', 'CHISQ.INV.RT', 'CHOOSE',
                'CHOOSE', 'CLEAN', 'CODE', 'COLUMN', 'COLUMNS', 'COMBIN',
                'COMBINA', 'COMPLEX', 'CONCATENATE', 'CONFIDENCE',
                'CONFIDENCE.NORM', 'CONFIDENCE.T', 'CONVERT', 'CORREL', 'COS',
                'COSH', 'COT', 'COTH', 'COUNT', 'COUNTA', 'COUNTBLANK',
                'COUNTIF', 'COUNTIFS', 'COUNTIN', 'COUNTUNIQUE', 'COVARIANCE.P',
                'COVARIANCE.S', 'CSC', 'CSCH', 'CUMIPMT', 'CUMPRINC', 'DATE',
                'DATEVALUE', 'DAY', 'DAYS', 'DAYS360', 'DB', 'DDB', 'DEC2BIN',
                'DEC2HEX', 'DEC2OCT', 'DECIMAL', 'DEGREES', 'DELTA', 'DEVSQ',
                'DIVIDE', 'DOLLAR', 'DOLLARDE', 'DOLLARFR', 'E', 'EDATE',
                'EFFECT', 'EOMONTH', 'EQ', 'ERF', 'ERFC', 'EVEN', 'EXACT',
                'EXP', 'EXPON.DIST', 'EXPONDIST', 'F.DIST', 'F.DIST.RT',
                'F.INV', 'F.INV.RT', 'FACT', 'FACTDOUBLE', 'FALSE', 'FDIST',
                'FDISTRT', 'FIND', 'FINV', 'FINVRT', 'FISHER', 'FISHERINV',
                'FIXED', 'FLATTEN', 'FLOOR', 'FORECAST', 'FREQUENCY', 'FV',
                'FVSCHEDULE', 'GAMMA', 'GAMMA.DIST', 'GAMMA.INV', 'GAMMADIST',
                'GAMMAINV', 'GAMMALN', 'GAMMALN.PRECISE', 'GAUSS', 'GCD',
                'GEOMEAN', 'GESTEP', 'GROWTH', 'GTE', 'HARMEAN', 'HEX2BIN',
                'HEX2DEC', 'HEX2OCT', 'HOUR', 'HTML2TEXT', 'HYPGEOM.DIST',
                'HYPGEOMDIST', 'IF', 'IMABS', 'IMAGINARY', 'IMARGUMENT',
                'IMCONJUGATE', 'IMCOS', 'IMCOSH', 'IMCOT', 'IMCSC', 'IMCSCH',
                'IMDIV', 'IMEXP', 'IMLN', 'IMLOG10', 'IMLOG2', 'IMPOWER',
                'IMPRODUCT', 'IMREAL', 'IMSEC', 'IMSECH', 'IMSIN', 'IMSINH',
                'IMSQRT', 'IMSUB', 'IMSUM', 'IMTAN', 'INT', 'INTERCEPT',
                'INTERVAL', 'IPMT', 'IRR', 'ISBINARY', 'ISBLANK', 'ISEVEN',
                'ISLOGICAL', 'ISNONTEXT', 'ISNUMBER', 'ISODD', 'ISODD',
                'ISOWEEKNUM', 'ISPMT', 'ISTEXT', 'JOIN', 'KURT', 'LARGE', 'LCM',
                'LEFT', 'LEN', 'LINEST', 'LN', 'LOG', 'LOG10', 'LOGEST',
                'LOGNORM.DIST', 'LOGNORM.INV', 'LOGNORMDIST', 'LOGNORMINV',
                'LOWER', 'LT', 'LTE', 'MATCH', 'MAX', 'MAXA', 'MEDIAN', 'MID',
                'MIN', 'MINA', 'MINUS', 'MINUTE', 'MIRR', 'MOD', 'MODE.MULT',
                'MODE.SNGL', 'MODEMULT', 'MODESNGL', 'MONTH', 'MROUND',
                'MULTINOMIAL', 'MULTIPLY', 'NE', 'NEGBINOM.DIST',
                'NEGBINOMDIST', 'NETWORKDAYS', 'NOMINAL', 'NORM.DIST',
                'NORM.INV', 'NORM.S.DIST', 'NORM.S.INV', 'NORMDIST', 'NORMINV',
                'NORMSDIST', 'NORMSINV', 'NOT', 'NOW', 'NPER', 'NPV', 'NUMBERS',
                'NUMERAL', 'OCT2BIN', 'OCT2DEC', 'OCT2HEX', 'ODD', 'OR',
                'PDURATION', 'PEARSON', 'PERCENTILEEXC', 'PERCENTILEINC',
                'PERCENTRANKEXC', 'PERCENTRANKINC', 'PERMUT', 'PERMUTATIONA',
                'PHI', 'PI', 'PMT', 'POISSON.DIST', 'POISSONDIST', 'POW',
                'POWER', 'PPMT', 'PROB', 'PRODUCT', 'PROPER', 'PV',
                'QUARTILE.EXC', 'QUARTILE.INC', 'QUARTILEEXC', 'QUARTILEINC',
                'QUOTIENT', 'RADIANS', 'RAND', 'RANDBETWEEN', 'RANK.AVG',
                'RANK.EQ', 'RANKAVG', 'RANKEQ', 'RATE', 'REFERENCE',
                'REGEXEXTRACT', 'REGEXMATCH', 'REGEXREPLACE', 'REPLACE', 'REPT',
                'RIGHT', 'ROMAN', 'ROUND', 'ROUNDDOWN', 'ROUNDUP', 'ROW',
                'ROWS', 'RRI', 'RSQ', 'SEARCH', 'SEC', 'SECH', 'SECOND',
                'SERIESSUM', 'SIGN', 'SIN', 'SINH', 'SKEW', 'SKEW.P', 'SKEWP',
                'SLN', 'SLOPE', 'SMALL', 'SPLIT', 'SPLIT', 'SQRT', 'SQRTPI',
                'STANDARDIZE', 'STDEV.P', 'STDEV.S', 'STDEVA', 'STDEVP',
                'STDEVPA', 'STDEVS', 'STEYX', 'SUBSTITUTE', 'SUBTOTAL', 'SUM',
                'SUMIF', 'SUMIFS', 'SUMPRODUCT', 'SUMSQ', 'SUMX2MY2',
                'SUMX2PY2', 'SUMXMY2', 'SWITCH', 'SYD', 'T', 'T.DIST',
                'T.DIST.2T', 'T.DIST.RT', 'T.INV', 'T.INV.2T', 'TAN', 'TANH',
                'TBILLEQ', 'TBILLPRICE', 'TBILLYIELD', 'TDIST', 'TDIST2T',
                'TDISTRT', 'TEXT', 'TIME', 'TIMEVALUE', 'TINV', 'TINV2T',
                'TODAY', 'TRANSPOSE', 'TREND', 'TRIM', 'TRIMMEAN', 'TRUE',
                'TRUNC', 'UNICHAR', 'UNICODE', 'UNIQUE', 'UPPER', 'VALUE',
                'VAR.P', 'VAR.S', 'VARA', 'VARP', 'VARPA', 'VARS', 'WEEKDAY',
                'WEEKNUM', 'WEIBULL.DIST', 'WEIBULLDIST', 'WORKDAY', 'XIRR',
                'XNPV', 'XOR', 'YEAR', 'YEARFRAC']
  excel_funs.sort(reverse=True)
  tokens = (
  'VARNAME', 'CONSTANT', 'STRING', 'FUNCTION', 'PLUS', 'MINUS', 'TIMES',
  'DIVIDE', 'GE', 'GT', 'LE', 'LT', 'EQ', 'NE','AND', 'OR', 'LPAREN', 'RPAREN',
  'BYROW')
  # Map var names
  t_VARNAME = r'\[.*?\]'
  t_CONSTANT = r'\d+'
  t_STRING = r'\"[a-zA-Z0-9]*\"'
  # Map operators and symbols
  t_PLUS = r'\+'
  t_MINUS = r'-'
  t_TIMES = r'\*'
  t_DIVIDE = r'/'
  t_LPAREN = r'\('
  t_RPAREN = r'\)'
  t_BYROW = r'\:\:'
  # Logical Operator
  t_GE = r'\>\='
  t_GT = r'\>'
  t_LE = r'\<\='
  t_LT = r'\<'
  t_EQ = r'\=\='
  t_NE = r'\<\>'
  t_AND = r'\&'
  t_OR = r'\|'
  # Map function use token decorator
  @staticmethod
  @lex.TOKEN('|'.join(excel_funs))
  def t_FUNCTION(t):
    return t

  # Ignored characters
  t_ignore = ','

  @staticmethod
  def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

  @staticmethod
  def t_error(t):
    # logging.error("Illegal character '%s'", t.value[0])
    t.lexer.skip(1)

  precedence = (('left', 'AND', 'OR'),
                ('left', 'GE', 'GT', 'LE', 'LT', 'EQ'),
                ('left', 'PLUS', 'MINUS'),
                ('left', 'TIMES', 'DIVIDE'),)

  # Expressions
  @staticmethod
  def p_expr(p):
    """expr : term
               | condition expr expr
               | expr PLUS expr
               | expr MINUS expr
               | expr TIMES expr
               | expr DIVIDE expr"""

    if len(p) > 2:
      # When writing functions, replace with the comments
      result, left, operator, right = p
      if operator == '+':
        result = left + right
      elif operator == '-':
        result = left - right
      elif operator == '*':
        result = left * right
      elif operator == '/':
        result = left / right
      else:
        result = {'condition': p[1], 'value1': p[2], 'value2': p[3]}
      p[0] = result
    else:
      p[0] = p[1]

  # Conditions
  @staticmethod
  def p_condition(p):
    """condition : expr GE expr
                      | expr GT expr
                      | expr LE expr
                      | expr LT expr
                      | expr EQ expr
                      | expr NE expr
                      | expr AND expr
                      | expr OR expr"""

    # logical
    #TODO put the operator options into dict outside funcs
    result, left, operator, right = p
    if operator == '>=':
      result = left >= right
    elif operator == '>':
      result = left > right
    elif operator == '<=':
      result = left <= right
    elif operator == '<':
      result = left < right
    elif operator == '==':
      result = left == right
    elif p[2] == '<>':
      result = left != right
    elif operator == '&':
      result = np.logical_and((left, right))
    elif operator == '|':
      result = np.logical_or((left, right))
    p[0] = result

  def p_function(self, p):
    """function : FUNCTION LPAREN expr RPAREN"""
    # p[1]: map the function name, p[3], map the insider
    try:
      p[0] = self.python_funcs[p[1]](p[3])  # p[0] = ('function', p[1], p[3])
    except KeyError:
      self.error_list.append("No matched function found for {}".format(
          p[1]))  # p[0] = self.python_funcs[p[1]](p[3])


  def p_function_bycol(self, p):
    """function_bycol : FUNCTION LPAREN varname BYROW RPAREN"""
    try:
      p[0] = self.python_funcs[p[1]](p[3], True)
    except KeyError:
      self.error_list.append("No matched function found for {}".format(
             p[1]))  # p[0] = self.python_funcs[p[1]](p[3])

  def p_varname(self, p):
    """varname : VARNAME"""
    # Replace the brackets
    p[1] = re.sub(r'[\[\]]', '', p[1])
    p[0] = self.assign(p[1])

  @staticmethod
  def p_constant(p):
    """constant : CONSTANT"""
    p[0] = float(p[1])

  @staticmethod
  def p_string(p):
    """string : STRING"""
    p[0] = p[1].replace('\"', '')

  @staticmethod
  def p_term(p):
    """term : varname
                | function
                | function_bycol
                | constant
                | string
                | varname term
                | function term
                | function_bycol term"""
    if len(p) > 2:
      # Combine columns
      p[0] = pd.concat([p[1], p[2]], axis=1)
    else:
      p[0] = p[1]

  @staticmethod
  def p_error(p):
    logging.error("Syntax error at '%s'", p)
