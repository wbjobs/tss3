import GUI from 'lil-gui';
import sphComputeShader from './shaders/sph_compute.wgsl?raw';
import particleRenderShader from './shaders/particle_render.wgsl?raw';
import obstacleRenderShader from './shaders/obstacle_render.wgsl?raw';

const MAX_PARTICLES = 8192;
const INITIAL_PARTICLES = 5000;
const GRID_RESOLUTION = 32;
const MAX_OBSTACLES = 64;

const params = {
    viscosity: 0.05,
    restDensity: 1.0,
    gasConstant: 200.0,
    gravity: -9.8,
    smoothingRadius: 0.025,
    dt: 0.0015,
    particleRadius: 0.015,
    particleCount: INITIAL_PARTICLES
};

const state = {
    device: null,
    context: null,
    format: null,
    particlesBuffer: null,
    gridIndicesBuffer: null,
    gridOffsetsBuffer: null,
    gridCountsBuffer: null,
    gridIndicesTempBuffer: null,
    paramsBuffer: null,
    obstaclesBuffer: null,
    renderUniformsBuffer: null,
    computeBindGroup: null,
    particleRenderBindGroup: null,
    obstacleRenderBindGroup: null,
    computePipelines: {},
    particleRenderPipeline: null,
    obstacleRenderPipeline: null,
    mouseActive: false,
    mouseMode: 0,
    mousePos: [0.5, 0.5],
    obstacles: [],
    activeParticleCount: INITIAL_PARTICLES,
    frameCount: 0,
    lastTime: performance.now(),
    fps: 0
};

const particleSize = 12 * 4;
const obstacleSize = 3 * 4;

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    console.error(message);
}

async function initWebGPU() {
    const canvas = document.getElementById('canvas');
    
    if (!navigator.gpu) {
        showError('您的浏览器不支持WebGPU。请使用最新版Chrome或Edge浏览器。');
        return false;
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        showError('无法获取WebGPU适配器。');
        return false;
    }
    
    const device = await adapter.requestDevice();
    if (!device) {
        showError('无法获取WebGPU设备。');
        return false;
    }
    
    device.addEventListener('uncapturederror', (event) => {
        console.error('WebGPU uncaptured error:', event.error);
        console.error('Message:', event.error.message);
    });
    
    state.device = device;
    
    const context = canvas.getContext('webgpu');
    if (!context) {
        showError('无法创建WebGPU上下文。');
        return false;
    }
    
    state.context = context;
    
    const format = navigator.gpu.getPreferredCanvasFormat();
    state.format = format;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    context.configure({
        device,
        format,
        alphaMode: 'premultiplied'
    });
    
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
    
    return true;
}

function createBuffers() {
    const device = state.device;
    
    const particlesData = new Float32Array(MAX_PARTICLES * 12);
    for (let i = 0; i < MAX_PARTICLES; i++) {
        if (i < INITIAL_PARTICLES) {
            const x = 0.3 + Math.random() * 0.4;
            const y = 0.3 + Math.random() * 0.5;
            particlesData[i * 12] = x;
            particlesData[i * 12 + 1] = y;
            particlesData[i * 12 + 2] = (Math.random() - 0.5) * 0.1;
            particlesData[i * 12 + 3] = (Math.random() - 0.5) * 0.1;
            particlesData[i * 12 + 4] = 1.0;
            particlesData[i * 12 + 5] = 0;
            particlesData[i * 12 + 6] = 0.2;
            particlesData[i * 12 + 7] = 0.6;
            particlesData[i * 12 + 8] = 1.0;
            particlesData[i * 12 + 9] = 0;
        } else {
            particlesData[i * 12 + 4] = -1;
        }
    }
    
    state.particlesBuffer = device.createBuffer({
        size: MAX_PARTICLES * particleSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(state.particlesBuffer.getMappedRange()).set(particlesData);
    state.particlesBuffer.unmap();
    
    state.gridIndicesBuffer = device.createBuffer({
        size: MAX_PARTICLES * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    state.gridOffsetsBuffer = device.createBuffer({
        size: GRID_RESOLUTION * GRID_RESOLUTION * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    state.gridCountsBuffer = device.createBuffer({
        size: GRID_RESOLUTION * GRID_RESOLUTION * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    state.gridIndicesTempBuffer = device.createBuffer({
        size: MAX_PARTICLES * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    state.paramsBuffer = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    state.obstaclesBuffer = device.createBuffer({
        size: MAX_OBSTACLES * obstacleSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    try {
        state.renderUniformsBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        console.log('renderUniformsBuffer created successfully');
    } catch (e) {
        console.error('Failed to create renderUniformsBuffer:', e);
    }
}

function updateParamsBuffer() {
    const device = state.device;
    const data = new ArrayBuffer(80);
    const view = new DataView(data);
    
    view.setUint32(0, state.activeParticleCount, true);
    view.setFloat32(4, params.smoothingRadius, true);
    view.setFloat32(8, params.smoothingRadius, true);
    view.setFloat32(12, params.restDensity, true);
    view.setFloat32(16, params.gasConstant, true);
    view.setFloat32(20, params.viscosity, true);
    view.setFloat32(24, params.dt, true);
    view.setFloat32(28, params.gravity, true);
    view.setFloat32(32, 0, true);
    view.setFloat32(36, 0, true);
    view.setFloat32(40, 1, true);
    view.setFloat32(44, 1, true);
    view.setFloat32(48, state.mousePos[0], true);
    view.setFloat32(52, state.mousePos[1], true);
    view.setUint32(56, state.mouseActive ? 1 : 0, true);
    view.setUint32(60, state.mouseMode, true);
    view.setUint32(64, GRID_RESOLUTION, true);
    view.setUint32(68, 0, true);
    view.setUint32(72, 0, true);
    
    device.queue.writeBuffer(state.paramsBuffer, 0, data);
}

// 检查并修正内核函数中的问题 - 确保密度计算正确
function checkParticleInitialization() {
    console.log('Initial particles:', state.activeParticleCount);
    console.log('Smoothing radius:', params.smoothingRadius);
    console.log('Grid size:', params.smoothingRadius);
}

function updateObstaclesBuffer() {
    const device = state.device;
    const data = new Float32Array(MAX_OBSTACLES * 3);
    
    for (let i = 0; i < MAX_OBSTACLES; i++) {
        if (i < state.obstacles.length) {
            data[i * 3] = state.obstacles[i].x;
            data[i * 3 + 1] = state.obstacles[i].y;
            data[i * 3 + 2] = state.obstacles[i].radius;
        } else {
            data[i * 3 + 2] = 0;
        }
    }
    
    device.queue.writeBuffer(state.obstaclesBuffer, 0, data);
}

function updateRenderUniforms() {
    const device = state.device;
    const canvas = document.getElementById('canvas');
    const data = new Float32Array(4);
    
    data[0] = canvas.width;
    data[1] = canvas.height;
    data[2] = params.particleRadius;
    data[3] = 0;
    
    device.queue.writeBuffer(state.renderUniformsBuffer, 0, data);
}

async function createComputePipelines() {
    const device = state.device;
    
    const computeModule = device.createShaderModule({
        code: sphComputeShader
    });
    
    const compilationInfo = await computeModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
        console.error('Compute shader compilation errors:');
        for (const msg of compilationInfo.messages) {
            console.error(`${msg.type}: ${msg.message}`);
        }
    }
    
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
        ]
    });
    
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });
    
    const entryPoints = ['buildGrid', 'clearGrid', 'countGridCells', 'buildGridOffsets', 'reorderParticles', 'clearGridCounts', 'computeDensityPressure', 'computeForces', 'integrate', 'addParticles'];
    
    for (const entry of entryPoints) {
        state.computePipelines[entry] = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: entry
            }
        });
    }
    
    console.log('Creating compute bind group with buffers:', {
        particles: state.particlesBuffer,
        gridIndices: state.gridIndicesBuffer,
        gridOffsets: state.gridOffsetsBuffer,
        params: state.paramsBuffer,
        obstacles: state.obstaclesBuffer
    });
    
    if (!state.particlesBuffer || !state.gridIndicesBuffer || !state.gridOffsetsBuffer || 
        !state.paramsBuffer || !state.obstaclesBuffer) {
        console.error('One or more buffers are null!');
        return;
    }
    
    state.computeBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: state.particlesBuffer } },
            { binding: 1, resource: { buffer: state.gridIndicesBuffer } },
            { binding: 2, resource: { buffer: state.gridOffsetsBuffer } },
            { binding: 3, resource: { buffer: state.paramsBuffer } },
            { binding: 4, resource: { buffer: state.obstaclesBuffer } },
            { binding: 5, resource: { buffer: state.gridCountsBuffer } },
            { binding: 6, resource: { buffer: state.gridIndicesTempBuffer } }
        ]
    });
    
    console.log('Compute bind group created successfully');
}

function createRenderPipelines() {
    const device = state.device;
    
    const particleModule = device.createShaderModule({
        code: particleRenderShader
    });
    
    const obstacleModule = device.createShaderModule({
        code: obstacleRenderShader
    });
    
    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
        ]
    });
    
    const renderPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [renderBindGroupLayout]
    });
    
    state.particleRenderPipeline = device.createRenderPipeline({
        layout: renderPipelineLayout,
        vertex: {
            module: particleModule,
            entryPoint: 'vs_main'
        },
        fragment: {
            module: particleModule,
            entryPoint: 'fs_main',
            targets: [{
                format: state.format,
                blend: {
                    color: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    },
                    alpha: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    }
                }
            }]
        },
        primitive: {
            topology: 'triangle-strip'
        }
    });
    
    state.obstacleRenderPipeline = device.createRenderPipeline({
        layout: renderPipelineLayout,
        vertex: {
            module: obstacleModule,
            entryPoint: 'vs_main'
        },
        fragment: {
            module: obstacleModule,
            entryPoint: 'fs_main',
            targets: [{
                format: state.format,
                blend: {
                    color: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    },
                    alpha: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    }
                }
            }]
        },
        primitive: {
            topology: 'triangle-strip'
        }
    });
    
    console.log('Creating render bind groups with buffers:', {
        renderUniforms: state.renderUniformsBuffer,
        particles: state.particlesBuffer,
        obstacles: state.obstaclesBuffer
    });
    
    if (!state.renderUniformsBuffer || !state.particlesBuffer || !state.obstaclesBuffer) {
        console.error('One or more render buffers are null!');
        return;
    }
    
    state.particleRenderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: state.renderUniformsBuffer } },
            { binding: 1, resource: { buffer: state.particlesBuffer } }
        ]
    });
    
    state.obstacleRenderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: state.renderUniformsBuffer } },
            { binding: 1, resource: { buffer: state.obstaclesBuffer } }
        ]
    });
    
    console.log('Render bind groups created successfully');
}

function dispatchCompute(encoder, pipeline, workgroups) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, state.computeBindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
}

function frame() {
    try {
        const device = state.device;
        const context = state.context;
        
        updateParamsBuffer();
        updateRenderUniforms();
        
        const particleWorkgroups = Math.ceil(state.activeParticleCount / 256);
        const gridWorkgroups = Math.ceil(GRID_RESOLUTION * GRID_RESOLUTION / 256);
        
        const encoder = device.createCommandEncoder();
        
        // 空间网格构建
        dispatchCompute(encoder, state.computePipelines.clearGrid, gridWorkgroups);
        dispatchCompute(encoder, state.computePipelines.countGridCells, particleWorkgroups);
        dispatchCompute(encoder, state.computePipelines.buildGridOffsets, 1);
        dispatchCompute(encoder, state.computePipelines.clearGridCounts, gridWorkgroups);
        dispatchCompute(encoder, state.computePipelines.reorderParticles, particleWorkgroups);
        
        // 物理计算 - 使用空间网格加速邻居查找
        dispatchCompute(encoder, state.computePipelines.computeDensityPressure, particleWorkgroups);
        dispatchCompute(encoder, state.computePipelines.computeForces, particleWorkgroups);
        dispatchCompute(encoder, state.computePipelines.integrate, particleWorkgroups);
        dispatchCompute(encoder, state.computePipelines.addParticles, particleWorkgroups);
        
        const view = context.getCurrentTexture().createView();
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view,
                clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });
        
        renderPass.setPipeline(state.particleRenderPipeline);
        renderPass.setBindGroup(0, state.particleRenderBindGroup);
        renderPass.draw(6, state.activeParticleCount);
        
        renderPass.setPipeline(state.obstacleRenderPipeline);
        renderPass.setBindGroup(0, state.obstacleRenderBindGroup);
        renderPass.draw(6, MAX_OBSTACLES);
        
        renderPass.end();
        
        device.queue.submit([encoder.finish()]);
        
        state.frameCount++;
        const now = performance.now();
        if (now - state.lastTime >= 1000) {
            state.fps = state.frameCount;
            state.frameCount = 0;
            state.lastTime = now;
            document.getElementById('fps').textContent = state.fps;
            document.getElementById('particleCount').textContent = state.activeParticleCount;
        }
        
        requestAnimationFrame(frame);
    } catch (e) {
        console.error('Error in frame:', e);
        console.error(e.stack);
        requestAnimationFrame(frame);
    }
}

function setupInput() {
    const canvas = document.getElementById('canvas');
    
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    
    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        state.mousePos[0] = (e.clientX - rect.left) / rect.width;
        state.mousePos[1] = 1.0 - (e.clientY - rect.top) / rect.height;
        state.mouseActive = true;
        
        if (e.button === 0) {
            state.mouseMode = 0;
            document.getElementById('mode').textContent = '添加粒子';
        } else if (e.button === 2) {
            state.mouseMode = 1;
            document.getElementById('mode').textContent = '添加障碍物';
            addObstacle(state.mousePos[0], state.mousePos[1], 0.05);
        }
    });
    
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        state.mousePos[0] = (e.clientX - rect.left) / rect.width;
        state.mousePos[1] = 1.0 - (e.clientY - rect.top) / rect.height;
        
        if (state.mouseActive && state.mouseMode === 0) {
            for (let i = 0; i < 10; i++) {
                activateParticle(state.mousePos[0], state.mousePos[1]);
            }
        } else if (state.mouseActive && state.mouseMode === 1) {
            addObstacle(state.mousePos[0], state.mousePos[1], 0.04);
        }
    });
    
    canvas.addEventListener('mouseup', () => {
        state.mouseActive = false;
    });
    
    canvas.addEventListener('mouseleave', () => {
        state.mouseActive = false;
    });
    
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
            e.preventDefault();
            state.obstacles = [];
            updateObstaclesBuffer();
        } else if (e.code === 'KeyR') {
            resetSimulation();
        }
    });
}

function activateParticle(x, y) {
    if (state.activeParticleCount >= MAX_PARTICLES) return;
    
    const device = state.device;
    const data = new Float32Array(12);
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * 0.02;
    
    data[0] = x + Math.cos(angle) * radius;
    data[1] = y + Math.sin(angle) * radius;
    data[2] = (Math.random() - 0.5) * 0.1;
    data[3] = (Math.random() - 0.5) * 0.1;
    data[4] = 0;
    data[5] = 0;
    data[6] = 0.2;
    data[7] = 0.5;
    data[8] = 1.0;
    data[9] = 0;
    
    const offset = state.activeParticleCount * particleSize;
    device.queue.writeBuffer(state.particlesBuffer, offset, data);
    state.activeParticleCount++;
}

function addObstacle(x, y, radius) {
    if (state.obstacles.length >= MAX_OBSTACLES) return;
    
    for (const obs of state.obstacles) {
        const dx = obs.x - x;
        const dy = obs.y - y;
        if (Math.sqrt(dx * dx + dy * dy) < obs.radius + radius * 0.5) {
            return;
        }
    }
    
    console.log('Adding obstacle at:', x, y, 'radius:', radius);
    state.obstacles.push({ x, y, radius });
    updateObstaclesBuffer();
    console.log('Total obstacles:', state.obstacles.length);
}

function resetSimulation() {
    const device = state.device;
    const particlesData = new Float32Array(MAX_PARTICLES * 12);
    
    for (let i = 0; i < MAX_PARTICLES; i++) {
        if (i < INITIAL_PARTICLES) {
            const x = 0.2 + Math.random() * 0.3;
            const y = 0.3 + Math.random() * 0.6;
            particlesData[i * 12] = x;
            particlesData[i * 12 + 1] = y;
            particlesData[i * 12 + 2] = (Math.random() - 0.5) * 0.5;
            particlesData[i * 12 + 3] = (Math.random() - 0.5) * 0.5;
            particlesData[i * 12 + 4] = 0;
            particlesData[i * 12 + 5] = 0;
            particlesData[i * 12 + 6] = 0.2;
            particlesData[i * 12 + 7] = 0.5;
            particlesData[i * 12 + 8] = 1.0;
            particlesData[i * 12 + 9] = 0;
        } else {
            particlesData[i * 12 + 4] = -1;
        }
    }
    
    device.queue.writeBuffer(state.particlesBuffer, 0, particlesData);
    state.activeParticleCount = INITIAL_PARTICLES;
    state.obstacles = [];
    updateObstaclesBuffer();
}

function setupGUI() {
    const gui = new GUI({ title: 'SPH 参数控制' });
    
    const simFolder = gui.addFolder('模拟参数');
    simFolder.add(params, 'viscosity', 0.0, 1.0, 0.01).name('粘度 (Viscosity)');
    simFolder.add(params, 'restDensity', 0.1, 5.0, 0.1).name('静止密度 (Rest Density)');
    simFolder.add(params, 'gasConstant', 50, 500, 10).name('气体常数 (Gas Constant)');
    simFolder.add(params, 'gravity', -20, 0, 0.5).name('重力 (Gravity)');
    simFolder.add(params, 'smoothingRadius', 0.01, 0.05, 0.001).name('光滑半径');
    simFolder.add(params, 'dt', 0.0005, 0.003, 0.0001).name('时间步长');
    
    const renderFolder = gui.addFolder('渲染参数');
    renderFolder.add(params, 'particleRadius', 0.003, 0.02, 0.001).name('粒子半径');
    
    const actions = {
        reset: () => resetSimulation(),
        clearObstacles: () => {
            state.obstacles = [];
            updateObstaclesBuffer();
        }
    };
    
    gui.add(actions, 'reset').name('重置模拟 (R)');
    gui.add(actions, 'clearObstacles').name('清除障碍物 (Space)');
}

async function main() {
    if (!await initWebGPU()) return;
    
    createBuffers();
    
    console.log('Buffers created:', {
        particles: !!state.particlesBuffer,
        gridIndices: !!state.gridIndicesBuffer,
        gridOffsets: !!state.gridOffsetsBuffer,
        params: !!state.paramsBuffer,
        obstacles: !!state.obstaclesBuffer,
        renderUniforms: !!state.renderUniformsBuffer
    });
    
    if (!state.renderUniformsBuffer) {
        console.error('renderUniformsBuffer is null!');
    }
    
    updateParamsBuffer();
    updateObstaclesBuffer();
    updateRenderUniforms();
    
    await createComputePipelines();
    console.log('Compute pipelines created:', Object.keys(state.computePipelines));
    
    createRenderPipelines();
    console.log('Render pipelines created');
    
    setupInput();
    setupGUI();
    
    console.log('Initial activeParticleCount:', state.activeParticleCount);
    console.log('INITIAL_PARTICLES:', INITIAL_PARTICLES);
    
    frame();
}

main().catch(console.error);
