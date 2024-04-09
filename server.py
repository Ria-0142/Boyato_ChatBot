from flask import Flask, render_template, request, jsonify
from waitress import serve
import ply.lex as lex
import ply.yacc as yacc
import sys
import os

reserved = {
    'print': 'PRINT',
    'for': 'FOR',
    'to': 'TO',
    'end': 'END',
    'while': 'WHILE',
    'do': 'DO',
    'True': 'TRUE',
    'False': 'FALSE',
    'not': 'NOT',
    'and': 'AND',
    'or': 'OR',
    'if': 'IF',
    'else': 'ELSE',
    'elif': 'ELIF',
    'func': 'FUNC',
    'call': 'CALL',
    'return': 'RETURN'
}

# List of token names
tokens = [
    'ID', 'INTEGER', 'FLOAT', 'STRING',
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
    'ASSIGN', 'INCREMENT', 'DECREMENT',
    'LPAREN', 'RPAREN', 'EXPO', 'GREATER_THAN',
    'LESS_THAN', 'EQUALS', 'GREATER_EQUAL',
    'LESS_EQUAL', 'MOD', 'PLUS_ASSIGN',
    'MINUS_ASSIGN', 'TIMES_ASSIGN',
    'DIVIDE_ASSIGN', 'MOD_ASSIGN', 'QUEST',
    'COLON', 'COMMA'
] + list(reserved.values())

# Regular expression rules for simple tokens
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_ASSIGN = r'='
t_INCREMENT = r'\+\+'
t_DECREMENT = r'--'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_EXPO = r'\*\*'
t_GREATER_THAN = r'\>'
t_LESS_THAN = r'\<'
t_EQUALS = r'\=='
t_GREATER_EQUAL = r'\<='
t_LESS_EQUAL = r'\>='
t_MOD = r'\%'
t_PLUS_ASSIGN = r'\+='
t_MINUS_ASSIGN = r'-='
t_TIMES_ASSIGN = r'\*='
t_DIVIDE_ASSIGN = r'/='
t_MOD_ASSIGN = r'\%='
t_QUEST = r'\?'
t_COLON = r'\:'
t_COMMA = r'\,'

# Define a rule so we can track line numbers


def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    return t

# Regular expression rule for identifiers


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')    # Check for reserved words
    return t

# Regular expression rule for floats


def t_FLOAT(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

# Regular expression rule for integers


def t_INTEGER(t):
    r'\d+'
    t.value = int(t.value)
    return t

# Regular expression rule for strings


def t_STRING(t):
    r'"[^"]*"'
    t.value = t.value[1:-1]  # Remove quotes
    return t


# Ignored characters (whitespace and newline)
t_ignore = ' \t\n'
t_ignore_COMMENT = r'\#.*'

# Error handling rule for newline


def t_error(t):
    raise SyntaxError(
        f"SyntaxError:Illegal character '{t.value[0]}' at line {t.lineno}")


# Build the lexer
lexer = lex.lex()

precedence = (
    ('nonassoc', 'LESS_THAN', 'GREATER_THAN', 'GREATER_EQUAL',
     'LESS_EQUAL', 'EQUALS'),  # Comparison operators
    ('left', 'AND'),  # Logical AND operator
    ('left', 'OR'),   # Logical OR operator
    ('left', 'PLUS', 'MINUS'),    # Addition and Subtraction
    ('left', 'TIMES', 'DIVIDE', 'MOD'),  # Multiplication, Division and Modulus
    ('right', 'EXPO'),    # Exponentiation operator
    ('right', 'UMINUS'),  # Unary minus operator
)

# Grammar rules


def p_program_eval(p):
    '''
    program_eval : program
    '''
    if p[1] is not None:
        result = None
        for statement in p[1]:
            ret_val = evaluate(statement, global_scope)
            if ret_val == '__return__':
                return global_scope.get('__return__')
            else:
                result = ret_val
        return result


def p_program(p):
    '''program : statements'''
    p[0] = p[1]


def p_statements(p):
    '''statements : statements statement
                  | statement'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

# def p_statement_declare_float(p):
#    '''statement : ID ASSIGN FLOAT'''
#    p[0] = ('=float', p[1], p[3])

# def p_statement_declare_int(p):
#    '''statement : ID ASSIGN INTEGER'''
#    p[0] = ('=int', p[1], p[3])


def p_statement_declare_string(p):
    '''statement : ID ASSIGN STRING'''
    p[0] = ('=string', p[1], p[3])


def p_statement_assign(p):
    '''statement : ID ASSIGN expression'''
    p[0] = ('=', p[1], p[3])


def p_statement_assign_function_call(p):
    '''statement : ID ASSIGN CALL ID LPAREN args RPAREN'''
    p[0] = ('assign_function_call', p[1], p[4], p[6])


def p_statement_return(p):
    '''statement : RETURN LPAREN expression RPAREN
                 | ID ASSIGN RETURN LPAREN expression RPAREN'''
    if len(p) == 5:
        p[0] = ('return', p[3])
    else:
        p[0] = ('return_assign', p[1], p[5])


def p_function_definition(p):
    '''statement : FUNC ID LPAREN params RPAREN COLON statements maybe_end'''
    p[0] = ('function_definition', p[2], p[4], p[7])


def p_params(p):
    '''params : params_list
              | empty'''
    p[0] = p[1]


def p_params_list(p):
    '''params_list : ID
                   | params_list COMMA ID'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_function_call(p):
    '''statement : CALL ID LPAREN args RPAREN'''
    p[0] = ('function_call', p[2], p[4])


def p_args(p):
    '''args : args_list
            | empty'''
    p[0] = p[1]


def p_args_list(p):
    '''args_list : expression
                 | args_list COMMA expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_statement_print(p):
    '''statement : PRINT LPAREN STRING RPAREN
                 | PRINT LPAREN expression RPAREN
                 | PRINT LPAREN STRING expression RPAREN'''
    if len(p) == 5:
        p[0] = ('print', p[3])
    elif len(p) == 6:
        if isinstance(p[4], str):
            p[0] = ('print', p[3] + " " + p[4])
        else:
            p[0] = ('print_with_variable', p[3], p[4])
    # elif len(p) == 4:
    #    p[0] = ('print', p[3])


def p_statement_compound_assign(p):
    '''statement : ID PLUS_ASSIGN expression
                 | ID MINUS_ASSIGN expression
                 | ID TIMES_ASSIGN expression
                 | ID DIVIDE_ASSIGN expression
                 | ID MOD_ASSIGN expression'''
    op = p[2]
    left = p[1]
    right = p[3]

    if op == '+=':
        p[0] = ('+=', left, right)
    elif op == '-=':
        p[0] = ('-=', left, right)
    elif op == '*=':
        p[0] = ('*=', left, right)
    elif op == '/=':
        p[0] = ('/=', left, right)
    elif op == '%=':
        p[0] = ('%=', left, right)


def p_statement_expr(p):
    '''statement : expression'''
    p[0] = p[1]


def p_expression_neg(p):
    '''
    expression : MINUS expression %prec UMINUS 
    '''
    p[0] = -p[2]


def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression EXPO expression
                  | expression MOD expression'''
    if p[2] == '+':
        p[0] = ('+', p[1], p[3])
    elif p[2] == '-':
        p[0] = ('-', p[1], p[3])
    elif p[2] == '*':
        p[0] = ('*', p[1], p[3])
    elif p[2] == '/':
        p[0] = ('/', p[1], p[3])
    elif p[2] == '**':
        p[0] = ('**', p[1], p[3])
    elif p[2] == '%':
        p[0] = ('%', p[1], p[3])


def p_expression_comparison(p):
    '''expression : expression LESS_THAN expression
                  | expression GREATER_THAN expression
                  | expression GREATER_EQUAL expression
                  | expression LESS_EQUAL expression
                  | expression EQUALS expression'''
    p[0] = (p[2], p[1], p[3])


def p_expression_logic_AND(p):
    '''expression : expression AND expression'''
    p[0] = ('AND', p[1], p[3])


def p_expression_logic_OR(p):
    '''expression : expression OR expression'''
    p[0] = ('OR', p[1], p[3])


def p_expression_logic_NOT(p):
    '''expression : NOT expression'''
    p[0] = ('NOT', p[2])


def p_expression_boolean(p):
    '''expression : TRUE
                  | FALSE'''
    p[0] = p[1]


def p_expression_group(p):
    '''expression : LPAREN expression RPAREN'''
    p[0] = p[2]


def p_expression_number(p):
    '''expression : INTEGER
                  | FLOAT'''
    p[0] = p[1]


def p_expression_id(p):
    '''expression : ID'''
    p[0] = ('id', p[1])


def p_expression_increment(p):
    '''expression : ID INCREMENT
                   | INCREMENT ID'''
    if p[1] == '++':
        p[0] = ('++', p[2])
    else:
        p[0] = ('++', p[1])


def p_expression_decrement(p):
    '''expression : ID DECREMENT'''
    if p[1] == '--':
        p[0] = ('--', p[2])
    else:
        p[0] = ('--', p[1])


def p_statement_if(p):
    '''statement : IF expression COLON statements maybe_end'''
    p[0] = ('if', p[2], p[4], p[5])


def p_statement_if_else(p):
    '''statement : IF expression COLON statements ELSE COLON statements maybe_end'''
    p[0] = ('if_else', p[2], p[4], p[7], p[8])


def p_statement_if_elif_else(p):
    '''statement : IF expression COLON statements elif_list ELSE COLON statements maybe_end'''
    p[0] = ('if_elif_else', p[2], p[4], p[5], p[8], p[9])


def p_elif_list(p):
    '''elif_list : ELIF expression COLON statements elif_list
                 | empty'''
    if len(p) == 2:
        p[0] = []
    elif len(p) == 6:
        p[0] = [(p[2], p[4])] + p[5]


def p_statement_for(p):
    '''statement : FOR ID ASSIGN expression TO expression statements maybe_end'''
    p[0] = ('for', p[2], p[4], p[6], p[7])


def p_statement_do_while(p):
    '''statement : DO statements WHILE expression maybe_end'''
    p[0] = ('do_while', p[2], p[4])


def p_statement_while(p):
    '''statement : WHILE expression statements maybe_end'''
    p[0] = ('while', p[2], p[3])


def p_expression_ternary(p):
    '''expression : expression QUEST expression COLON expression'''
    p[0] = ('?', p[1], p[3], p[5])


def p_maybe_end(p):
    '''maybe_end :
                | END'''
    pass


def p_empty(p):
    '''empty :'''
    pass

# Error handling rule


def p_error(p):
    if p:
        print(f"Syntax error: Unexpected token '{p.value}' at line {p.lineno}")
        raise SyntaxError(
            f"Syntax error: Unexpected token '{p.value}' at line {p.lineno}")
    else:
        print("Syntax error: Unexpected end of input")
        raise SyntaxError("Syntax error: Unexpected end of input")


# Build the parser
parser = yacc.yacc()

# variables = {}


class Scope:
    def __init__(self, parent=None):
        self.variables = {}
        self.parent = parent

    def get(self, name):
        if name in self.variables:
            return self.variables[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise NameError(f"Undefined variable '{name}'")

    def set(self, name, value):
        self.variables[name] = value

    def define(self, name):
        self.variables[name] = None

    def new_child(self):
        return Scope(self)


global_scope = Scope()


def evaluate(node, scope):
    if isinstance(node, (int, float, str)):
        return node
    elif isinstance(node, tuple):
        op = node[0]

        if op in ['+', '-', '*', '/', '**', '%']:
            left, right = node[1:]
            left_val = evaluate(left, scope)
            right_val = evaluate(right, scope)
            if None in [left_val, right_val]:
                return False  # Treat None as False
            if isinstance(left_val, str) or isinstance(right_val, str):
                print(
                    "Error: Cannot compare strings with '+', '-', '*', '/' ,'%' or '**'")
                return False
            if left_val is None or right_val is None:
                return None
            if op == '+':
                return evaluate(left, scope) + evaluate(right, scope)
            elif op == '-':
                return evaluate(left, scope) - evaluate(right, scope)
            elif op == '*':
                return evaluate(left, scope) * evaluate(right, scope)
            elif op == '/':
                left_val = evaluate(node[1], scope)
                right_val = evaluate(node[2], scope)
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return left_val / right_val
            elif op == '%':
                left_val = evaluate(node[1], scope)
                right_val = evaluate(node[2], scope)
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return left_val % right_val
            elif op == '**':
                return evaluate(left, scope) ** evaluate(right, scope)
        elif op in ['<', '>', '==', '<=', '>=']:
            left, right = node[1:]
            left_val = evaluate(left, scope)
            right_val = evaluate(right, scope)
            if None in [left_val, right_val]:
                return False  # Treat None as False
            if isinstance(left_val, str) or isinstance(right_val, str):
                print("Error: Cannot compare strings with '<', '>','<=','>=' or '=='")
                return False
            if op == '<':
                return left_val < right_val
            elif op == '>':
                return left_val > right_val
            if op == '<=':
                return left_val <= right_val
            elif op == '>=':
                return left_val >= right_val
            elif op == '==':
                return left_val == right_val
        elif op == 'AND':
            left_val = evaluate(node[1], scope)
            right_val = evaluate(node[2], scope)
            return left_val and right_val
        elif op == 'OR':
            left_val = evaluate(node[1], scope)
            right_val = evaluate(node[2], scope)
            return left_val or right_val
        elif op == 'NOT':
            # Handle logical NOT operation
            return not evaluate(node[1], scope)
        elif op in ['TRUE', 'FALSE']:
            # Return boolean literals
            return op == 'TRUE'
        elif op == '=':
            left, right = node[1:]
            value = evaluate(right, scope)
            scope.set(left, value)  # Set the variable in the current scope
            return value
        elif op == '+=':
            var_name = node[1]
            value = evaluate(node[2], scope)
            if value is None:
                return None
            if var_name in global_scope.variables:
                # Increment the variable in the global scope
                current_value = global_scope.get(var_name)
                new_value = current_value + value
                global_scope.set(var_name, new_value)
                return new_value  # Return the new value after the compound assignment
            else:
                raise NameError(f"Undefined variable '{var_name}'")
        elif op == '-=':
            var_name = node[1]
            value = evaluate(node[2], scope)
            if value is None:
                return None
            if var_name in global_scope.variables:
                # Increment the variable in the global scope
                current_value = global_scope.get(var_name)
                new_value = current_value - value
                global_scope.set(var_name, new_value)
                return new_value  # Return the new value after the compound assignment
            else:
                raise NameError(f"Undefined variable '{var_name}'")
        elif op == '*=':
            var_name = node[1]
            value = evaluate(node[2], scope)
            if value is None:
                return None
            if var_name in global_scope.variables:
                # Increment the variable in the global scope
                current_value = global_scope.get(var_name)
                new_value = current_value * value
                global_scope.set(var_name, new_value)
                return new_value  # Return the new value after the compound assignment
            else:
                raise NameError(f"Undefined variable '{var_name}'")
        elif op == '/=':
            var_name = node[1]
            value = evaluate(node[2], scope)
            if value is None:
                return None
            if var_name in global_scope.variables:
                current_value = global_scope.get(var_name)
                if value == 0:
                    raise ZeroDivisionError("Division by zero")
                new_value = current_value / value
                global_scope.set(var_name, new_value)
                return new_value  # Return the new value after the compound assignment
            else:
                raise NameError(f"Undefined variable '{var_name}'")
        elif op == '%=':
            var_name = node[1]
            value = evaluate(node[2], scope)
            if value is None:
                return None
            if var_name in global_scope.variables:
                current_value = global_scope.get(var_name)
                if value == 0:
                    raise ZeroDivisionError("Division by zero")
                new_value = current_value % value
                global_scope.set(var_name, new_value)
                return new_value  # Return the new value after the compound assignment
            else:
                raise NameError(f"Undefined variable '{var_name}'")
        elif op == '=string':
            left, right = node[1:]
            value = str(right)
            scope.set(left, value)
            return value
        elif op == 'id':
            left = node[1]
            if left is not None:
                return scope.get(left)
        elif op == '++':
            var_name = node[1]
            value = scope.get(var_name)
            if value is None:
                # print(f"Error: Undefined variable '{var_name}'")
                return None

            if var_name in scope.variables:
                # Increment variable in the current scope
                current_value = scope.get(var_name)
                new_value = current_value + 1
                scope.set(var_name, new_value)
            else:
                # Variable doesn't exist in the current scope, try to find it in parent scopes
                parent_scope = scope.parent
                while parent_scope:
                    if var_name in parent_scope.variables:
                        current_value = parent_scope.get(var_name)
                        new_value = current_value + 1
                        parent_scope.set(var_name, new_value)
                        break
                    parent_scope = parent_scope.parent
            return new_value  # Return the new value after incrementing
        elif op == '--':
            var_name = node[1]
            value = scope.get(var_name)
            if value is None:
                # print(f"Error: Undefined variable '{var_name}'")
                return None

            if var_name in scope.variables:
                # Increment variable in the current scope
                current_value = scope.get(var_name)
                new_value = current_value - 1
                scope.set(var_name, new_value)
            else:
                # Variable doesn't exist in the current scope, try to find it in parent scopes
                parent_scope = scope.parent
                while parent_scope:
                    if var_name in parent_scope.variables:
                        current_value = parent_scope.get(var_name)
                        new_value = current_value - 1
                        parent_scope.set(var_name, new_value)
                        break
                    parent_scope = parent_scope.parent
        elif op == 'print':
            if isinstance(node[1], str):  # Check if it's a string literal
                output_values.append(node[1])
            else:
                result = evaluate(node[1], scope)
                if result is not None:
                    output_values.append(str(result))
                else:
                    return None
        elif op == 'print_with_variable':
            string_literal = node[1]
            variable_value = evaluate(node[2], scope)
            if variable_value is not None:
                print(string_literal, variable_value)
                return string_literal + " " + str(variable_value)
        elif op == 'if':
            condition = node[1]
            if evaluate(condition, scope):
                new_scope = scope.new_child()
                statements = node[2]
                for statement in statements:
                    evaluate(statement, new_scope)
        elif op == 'if_else':
            condition = node[1]
            if evaluate(condition, scope):
                new_scope = scope.new_child()
                statements = node[2]
                for statement in statements:
                    evaluate(statement, new_scope)
            else:
                new_scope = scope.new_child()
                statements = node[3]
                for statement in statements:
                    evaluate(statement, new_scope)
        elif op == 'if_elif_else':
            if_block_condition = node[1]
            if_block_statements = node[2]
            elif_conditions_statements = node[3]
            else_block_statements = node[4]

            if evaluate(if_block_condition, scope):
                new_scope = scope.new_child()
                for statement in if_block_statements:
                    evaluate(statement, new_scope)
            else:
                for condition, statements in elif_conditions_statements:
                    if evaluate(condition, scope):
                        new_scope = scope.new_child()
                        for statement in statements:
                            evaluate(statement, new_scope)
                        break
                    else:
                        new_scope = scope.new_child()
                        for statement in else_block_statements:
                            evaluate(statement, new_scope)
        elif op == 'for':
            loop_var = node[1]
            start_val = evaluate(node[2], scope)
            end_val = evaluate(node[3], scope)
            loop_statements = node[4]
            if all(val is not None for val in [start_val, end_val]):
                for i in range(start_val, end_val + 1):
                    new_scope = scope.new_child()
                    new_scope.set(loop_var, i)
                    for statement in loop_statements:
                        evaluate(statement, new_scope)

        elif op == 'do_while':
            loop_statements = node[1]
            loop_condition = node[2]
            new_scope = scope.new_child()

            # Execute the loop body at least once
            while True:
                for statement in loop_statements:
                    evaluate(statement, new_scope)
                # Evaluate the loop condition
                condition_result = evaluate(loop_condition, scope)
                if not condition_result:
                    break

        elif op == 'while':
            condition = node[1]
            loop_statements = node[2]
            new_scope = scope.new_child()
            while evaluate(condition, scope):
                for statement in loop_statements:
                    evaluate(statement, new_scope)

        elif op == '?':
            condition = evaluate(node[1], scope)
            true_case = evaluate(node[2], scope)
            false_case = evaluate(node[3], scope)
            return true_case if condition else false_case

        elif op == 'return':
            value = evaluate(node[1], scope)
            # Store the return value in the function scope
            scope.set('__return__', value)
            return '__return__'
        elif op == 'return_assign':
            value = evaluate(node[2], scope)
            variable_name = node[1]
            scope.set(variable_name, value)
            # Store the return value in the function scope
            scope.set('__return__', value)
            return '__return__'
        elif op == 'function_definition':
            # Store the function definition in the current scope
            function_name = node[1]
            params = node[2]
            statements = node[3]
            scope.set(function_name, ('function', params, statements))

        elif op == 'function_call':
            # Execute the function call in a new local scope
            function_name = node[1]
            args = node[2]
            function_definition = scope.get(function_name)
            if function_definition:
                params, statements = function_definition[1], function_definition[2]
                # Create a new scope with the current scope as parent
                new_scope = Scope(parent=scope)
                for param, arg in zip(params, args):
                    new_scope.set(param, evaluate(arg, scope))
                for statement in statements:
                    evaluate(statement, new_scope)
        elif op == 'assign_function_call':
            var_name = node[1]
            function_name = node[2]
            args = node[3]
            function_definition = scope.get(function_name)
            if function_definition:
                params, statements = function_definition[1], function_definition[2]
                # Create a new scope with the current scope as parent
                new_scope = Scope(parent=scope)
                for param, arg in zip(params, args):
                    new_scope.set(param, evaluate(arg, scope))
                for statement in statements:
                    evaluate(statement, new_scope)
                # Get the return value from the function scope
                return_value = new_scope.get('__return__')
                # Assign the return value to the variable in the current scope
                scope.set(var_name, return_value)
                return return_value
    else:
        raise ValueError(f"Invalid expression '{node}'")


def execute_code(t_code):
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Prompt the user for a file path or use a default path
        # file_path = input("Enter the file path: ")
        # or
        file_path = "source_file.txt"

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            source_code = file.read()

        try:
            # global_scope = Scope()
            result = parser.parse(t_code)
            return result
        except NameError as e:
            return f"Error: {e}"
            # sys.exit(1)
        except ZeroDivisionError as e:
            return f"Error: {e}"
            # sys.exit(1)
        except Exception as e:
            return f"Error: {e}"
            # sys.exit(1)


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/execute', methods=['POST'])
def execute():
    code = request.json.get('code')
    # print(code)
    if code:
        global output_values
        output_values = []
        output = execute_code(code)

        if output is not None:
            return jsonify({'output': output})

        output_text = '\n'.join(map(str, output_values))
        return jsonify({'output': output_text})
    return jsonify({'output': 'No code provided'})


if __name__ == '__main__':
    serve(app,host="0.0.0.0",port=8000)
