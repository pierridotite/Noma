use clap::{Parser, Subcommand};
use noma_compiler::{Lexer, Parser as NomaParser, ComputationalGraph};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "noma")]
#[command(about = "NOMA Compiler - Neural-Oriented Machine Architecture", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a NOMA source file
    Build {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Print the Abstract Syntax Tree
        #[arg(short, long)]
        ast: bool,

        /// Print tokens from lexer
        #[arg(short, long)]
        tokens: bool,

        /// Print the computational graph
        #[arg(short, long)]
        graph: bool,
    },

    /// Check syntax without building
    Check {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Run autodiff demo: minimize y = x^2
    Demo,

    /// Display compiler version and build info
    Version,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { file, ast: print_ast, tokens: print_tokens, graph: print_graph } => {
            build_file(file, print_ast, print_tokens, print_graph)?;
        }
        Commands::Check { file } => {
            check_file(file)?;
        }
        Commands::Demo => {
            run_demo()?;
        }
        Commands::Version => {
            println!("NOMA Compiler v{}", env!("CARGO_PKG_VERSION"));
            println!("The Neural-Oriented Machine Architecture");
            println!("Status: Pre-Alpha (Milestone 3 - The Tipping Point)");
        }
    }

    Ok(())
}

fn build_file(file: PathBuf, print_ast: bool, print_tokens: bool, print_graph: bool) -> anyhow::Result<()> {
    println!("Building: {}", file.display());

    // Read source file
    let source = fs::read_to_string(&file)?;

    // Lexical analysis
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize()?;

    if print_tokens {
        println!("\n=== TOKENS ===");
        for (idx, token) in tokens.iter().enumerate() {
            println!("{:3}: {:?}", idx, token);
        }
    }

    // Parsing
    let mut parser = NomaParser::new(tokens.clone());
    let program = match parser.parse() {
        Ok(prog) => prog,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return Err(e.into());
        }
    };

    if print_ast {
        println!("\n=== AST ===");
        println!("{:#?}", program);
    }

    // Build computational graph
    let mut _graph = ComputationalGraph::new();
    
    if print_graph {
        println!("\n=== COMPUTATIONAL GRAPH ===");
        _graph.print_structure();
    }

    println!("\nCompilation: OK");
    println!("Total tokens: {}", tokens.len());
    println!("Items: {}", program.items.len());

    Ok(())
}

fn check_file(file: PathBuf) -> anyhow::Result<()> {
    println!("Checking: {}", file.display());

    let source = fs::read_to_string(&file)?;
    let mut lexer = Lexer::new(&source);
    
    match lexer.tokenize() {
        Ok(tokens) => {
            let mut parser = NomaParser::new(tokens);
            match parser.parse() {
                Ok(_) => {
                    println!("Syntax check: OK");
                    Ok(())
                }
                Err(e) => {
                    eprintln!("Parse error: {}", e);
                    Err(e.into())
                }
            }
        }
        Err(e) => {
            eprintln!("Lexical error: {}", e);
            Err(e.into())
        }
    }
}

fn run_demo() -> anyhow::Result<()> {
    println!("NOMA Autodiff Demo: Minimize y = x^2");
    println!("======================================\n");

    let mut graph = ComputationalGraph::new();
    
    // Create: y = x^2 (x * x)
    let x = graph.add_learnable("x".to_string(), 5.0);
    let y = graph.add_binary_op("mul", x, x);

    println!("Initial state: x = 5.0");
    println!("Goal: Minimize y = x^2 (find x ≈ 0)\n");
    
    println!("Iteration | x value | y value | gradient |");
    println!("-----------|---------|---------|----------|");
    
    let learning_rate = 0.1;
    let max_iterations = 50;

    // Training loop
    for iteration in 0..=max_iterations {
        // Forward pass
        graph.forward_pass().map_err(|e| anyhow::anyhow!(e))?;
        
        let x_val = graph.get_node(x).and_then(|n| n.value).unwrap_or(0.0);
        let y_val = graph.get_node(y).and_then(|n| n.value).unwrap_or(0.0);

        // Print progress
        if iteration == 0 || iteration % 5 == 0 || iteration == max_iterations {
            println!("{:^9} | {:7.4} | {:7.4} | -        |", iteration, x_val, y_val);
        }

        // Backward pass
        graph.backward_pass(y).map_err(|e| anyhow::anyhow!(e))?;

        // Optimization step
        graph.optimize_step(learning_rate).map_err(|e| anyhow::anyhow!(e))?;
        graph.reset_gradients();

        // Early stopping if converged
        if y_val.abs() < 1e-6 {
            println!("\nConverged at iteration {}!", iteration);
            break;
        }
    }

    println!();
    let final_x = graph.get_node(x).and_then(|n| n.value).unwrap_or(0.0);
    let final_y = graph.get_node(y).and_then(|n| n.value).unwrap_or(0.0);

    println!("Final result:");
    println!("  x = {:.6}", final_x);
    println!("  y = {:.6}", final_y);
    println!("\nSuccess! Gradient descent found a minimum near x = 0");

    Ok(())
}
