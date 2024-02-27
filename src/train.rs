

fn train<P: AsRef<Path>>(save_dir: P) -> Result<(), Box<dyn Error>> {
    // ================
    // Build the model.
    // ================
    let mut scope = Scope::new_root_scope();
    let scope = &mut scope;
    // Size of the hidden layer.
    // This is far more than is necessary, but makes it train more reliably.
    let hidden_size: u64 = 8;
    let input = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape([1u64, 2])
        .build(&mut scope.with_op_name("input"))?;
    let label = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape([1u64])
        .build(&mut scope.with_op_name("label"))?;
    // Hidden layer.
    let (vars1, layer1) = layer(
        input.clone(),
        2,
        hidden_size,
        &|x, scope| Ok(ops::tanh(x, scope)?.into()),
        scope,
    )?;
    // Output layer.
    let (vars2, layer2) = layer(layer1.clone(), hidden_size, 1, &|x, _| Ok(x), scope)?;
    let error = ops::sub(layer2.clone(), label.clone(), scope)?;
    let error_squared = ops::mul(error.clone(), error, scope)?;
    let mut optimizer = AdadeltaOptimizer::new();
    optimizer.set_learning_rate(ops::constant(1.0f32, scope)?);
    let mut variables = Vec::new();
    variables.extend(vars1);
    variables.extend(vars2);
    let (minimizer_vars, minimize) = optimizer.minimize(
        scope,
        error_squared.clone().into(),
        MinimizeOptions::default().with_variables(&variables),
    )?;
