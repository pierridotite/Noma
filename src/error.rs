use thiserror::Error;

#[derive(Error, Debug)]
pub enum NomaError {
    #[error("Lexical error at line {line}, column {column}: {message}")]
    LexError {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("Unexpected character '{ch}' at line {line}, column {column}")]
    UnexpectedCharacter {
        ch: char,
        line: usize,
        column: usize,
    },

    #[error("Unterminated string at line {line}")]
    UnterminatedString { line: usize },

    #[error("Invalid number format at line {line}, column {column}")]
    InvalidNumber { line: usize, column: usize },
}

impl NomaError {
    pub fn lex_error(message: impl Into<String>, line: usize, column: usize) -> Self {
        NomaError::LexError {
            message: message.into(),
            line,
            column,
        }
    }

    pub fn unexpected_char(ch: char, line: usize, column: usize) -> Self {
        NomaError::UnexpectedCharacter { ch, line, column }
    }
}
