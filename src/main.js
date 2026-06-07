import GUI from 'lil-gui';
import sphComputeShader from './shaders/sph_compute.wgsl?raw';
import particleRenderShader from './shaders/particle_render.wgsl?raw';
import obstacleRenderShader from './shaders/obstacle_render.wgsl?raw';
import heightfieldRenderShader from './shaders/heightfield_render.wgsl?raw';

const MAX_PARTICLES = 32768;
const INITIAL_PARTICLES = 20000;
const GRID_RESOLUTION = 64;
const MAX_OBSTACLES = 64;
const HEIGHTFIELD_RESOLUTION = 128;
const SPLASH_THRESHOLD = 3.0;
const SPLASH_PARTICLE_COUNT = 3;

const params = {
    viscosity: 0.05,
    restDensity: 1.0,
    gasConstant: 200.0,
    gravity: -9.8,
    smoothingRadius: 0.025,
    dt: 0.0015,
    particleRadius: 0.015,
    particleCount: INITIAL_PARTICLES,
    rotationSpeed: 2.0,
    erosionStrength: 0.1,
    depositionStrength: 0.05,
    splashStrength: 0.5
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
    heightfieldBuffer: null,
    heightfieldRenderBuffer: null,
    paramsBuffer: null,
    obstaclesBuffer: null,
    renderUniformsBuffer: null,
    computeBindGroup: null,
    heightfieldBindGroup: null,
    particleRenderBindGroup: null,
    obstacleRenderBindGroup: null,
    heightfieldRenderBindGroup: null,
    computePipelines: {},
    particleRenderPipeline: null,
    obstacleRenderPipeline: null,
    heightfieldRenderPipeline: null,
    mouseActive: false,
    mouseMode: 0,
    mousePos: [0.5, 0.5],
    obstacles: [],
    activeParticleCount: INITIAL_PARTICLES,
    frameCount: 0,
    lastTime: performance.now(),
    fps: 0,
    querySet: null,
    queryBuffer: null,
    queryResolveBuffer: null,
    timestamps: {},
    memoryUsage: {},
    computePassTimings: {}
};

const particleSize = 12 * 4;
const obstacleSize = 5 * 4;

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
    
    const requiredFeatures = [];
    if (adapter.features.has('timestamp-query')) {
        requiredFeatures.push('timestamp-query');
    }
    
    const requiredLimits = {};
    if (adapter.limits.maxComputeWorkgroupStorageSize >= 32768) {
        requiredLimits.maxComputeWorkgroupStorageSize = 32768;
    }
    
    const device = await adapter.requestDevice({
        requiredFeatures,
        requiredLimits
    });
    if (!device) {
        showError('无法获取WebGPU设备。');
        return false;
    }
    
    state.supportsTimestampQuery = device.features.has('timestamp-query');
    state.timestampApiAvailable = state.supportsTimestampQuery && typeof device.createCommandEncoder().writeTimestamp === 'function';
    
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
    
    state.heightfieldBuffer = device.createBuffer({
        size: HEIGHTFIELD_RESOLUTION * HEIGHTFIELD_RESOLUTION * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    state.heightfieldRenderBuffer = device.createBuffer({
        size: HEIGHTFIELD_RESOLUTION * HEIGHTFIELD_RESOLUTION * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    
    const heightfieldData = new Float32Array(HEIGHTFIELD_RESOLUTION * HEIGHTFIELD_RESOLUTION);
    for (let y = 0; y < HEIGHTFIELD_RESOLUTION; y++) {
        for (let x = 0; x < HEIGHTFIELD_RESOLUTION; x++) {
            const nx = x / HEIGHTFIELD_RESOLUTION;
            const ny = y / HEIGHTFIELD_RESOLUTION;
            let height = 0.1 * (1.0 - ny);
            if (nx > 0.3 && nx < 0.7 && ny > 0.2 && ny < 0.3) {
                height += 0.1;
            }
            heightfieldData[y * HEIGHTFIELD_RESOLUTION + x] = height;
        }
    }
    device.queue.writeBuffer(state.heightfieldBuffer, 0, heightfieldData);
    device.queue.writeBuffer(state.heightfieldRenderBuffer, 0, heightfieldData);
    
    state.paramsBuffer = device.createBuffer({
        size: 96,
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
    
    if (state.supportsTimestampQuery) {
        const queryCount = 32;
        state.querySet = device.createQuerySet({
            type: 'timestamp',
            count: queryCount
        });
        
        state.queryBuffer = device.createBuffer({
            size: queryCount * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        state.queryResolveBuffer = device.createBuffer({
            size: queryCount * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
    } else {
        console.warn('Timestamp queries not supported. Performance metrics will be limited.');
    }
}

function updateParamsBuffer() {
    const device = state.device;
    const data = new ArrayBuffer(96);
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
    view.setUint32(68, MAX_PARTICLES, true);
    view.setUint32(72, HEIGHTFIELD_RESOLUTION, true);
    view.setFloat32(76, params.erosionStrength, true);
    view.setFloat32(80, params.depositionStrength, true);
    view.setFloat32(84, params.splashStrength, true);
    view.setFloat32(88, params.rotationSpeed, true);
    view.setUint32(92, 0, true);
    
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
    const data = new Float32Array(MAX_OBSTACLES * 5);
    
    for (let i = 0; i < MAX_OBSTACLES; i++) {
        if (i < state.obstacles.length) {
            data[i * 5] = state.obstacles[i].x;
            data[i * 5 + 1] = state.obstacles[i].y;
            data[i * 5 + 2] = state.obstacles[i].radius;
            data[i * 5 + 3] = state.obstacles[i].rotation || 0;
            data[i * 5 + 4] = state.obstacles[i].angularVelocity || 0;
        } else {
            data[i * 5 + 2] = 0;
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
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
        ]
    });
    
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });
    
    const entryPoints = ['clearGrid', 'countGridCells', 'buildGridOffsets', 'reorderParticles', 'clearGridCounts', 'computeDensityPressure', 'computeForces', 'integrate', 'addParticles', 'updateObstacles', 'erodeHeightfield'];
    
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
        obstacles: state.obstaclesBuffer,
        heightfield: state.heightfieldBuffer
    });
    
    if (!state.particlesBuffer || !state.gridIndicesBuffer || !state.gridOffsetsBuffer || 
        !state.paramsBuffer || !state.obstaclesBuffer || !state.heightfieldBuffer) {
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
            { binding: 6, resource: { buffer: state.gridIndicesTempBuffer } },
            { binding: 7, resource: { buffer: state.heightfieldBuffer } }
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
    
    const heightfieldModule = device.createShaderModule({
        code: heightfieldRenderShader
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
    
    state.heightfieldRenderPipeline = device.createRenderPipeline({
        layout: renderPipelineLayout,
        vertex: {
            module: heightfieldModule,
            entryPoint: 'vs_main'
        },
        fragment: {
            module: heightfieldModule,
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
    
    state.heightfieldRenderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: state.renderUniformsBuffer } },
            { binding: 1, resource: { buffer: state.heightfieldRenderBuffer } }
        ]
    });
    
    console.log('Render bind groups created successfully');
}

let queryIndex = 0;
const computePassNames = ['updateObstacles', 'clearGrid', 'countGridCells', 'buildGridOffsets', 'clearGridCounts', 'reorderParticles', 'computeDensityPressure', 'computeForces', 'integrate', 'addParticles', 'erodeHeightfield'];

function dispatchCompute(encoder, pipeline, workgroups) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, state.computeBindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
}

function dispatchComputeWithQuery(encoder, pipeline, workgroups, queryIdx) {
    if (state.supportsTimestampQuery && state.querySet && state.timestampApiAvailable) {
        try {
            encoder.writeTimestamp(state.querySet, queryIdx * 2);
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, state.computeBindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
            encoder.writeTimestamp(state.querySet, queryIdx * 2 + 1);
        } catch (e) {
            console.warn('Timestamp write failed, disabling timestamp queries:', e.message);
            state.timestampApiAvailable = false;
            dispatchCompute(encoder, pipeline, workgroups);
        }
    } else {
        dispatchCompute(encoder, pipeline, workgroups);
    }
}

async function readTimestamps() {
    try {
        if (!state.queryResolveBuffer) return;
        
        await state.queryResolveBuffer.mapAsync(GPUMapMode.READ);
        const data = new BigUint64Array(state.queryResolveBuffer.getMappedRange());
        
        const timestamps = {};
        for (let i = 0; i < computePassNames.length; i++) {
            const start = data[i * 2];
            const end = data[i * 2 + 1];
            if (start && end) {
                const duration = Number(end - start) / 1000000;
                timestamps[computePassNames[i]] = duration;
                state.computePassTimings[computePassNames[i]] = duration;
            }
        }
        
        state.queryResolveBuffer.unmap();
        
        updatePerformancePanel();
    } catch (e) {
        console.warn('Failed to read timestamps:', e);
    }
}

function calculateMemoryUsage() {
    const buffers = [
        { name: 'particles', buffer: state.particlesBuffer },
        { name: 'gridIndices', buffer: state.gridIndicesBuffer },
        { name: 'gridOffsets', buffer: state.gridOffsetsBuffer },
        { name: 'gridCounts', buffer: state.gridCountsBuffer },
        { name: 'gridIndicesTemp', buffer: state.gridIndicesTempBuffer },
        { name: 'heightfield', buffer: state.heightfieldBuffer },
        { name: 'heightfieldRender', buffer: state.heightfieldRenderBuffer },
        { name: 'params', buffer: state.paramsBuffer },
        { name: 'obstacles', buffer: state.obstaclesBuffer },
        { name: 'renderUniforms', buffer: state.renderUniformsBuffer }
    ];
    
    let total = 0;
    const usage = {};
    for (const { name, buffer } of buffers) {
        if (buffer) {
            const size = buffer.size;
            usage[name] = size;
            total += size;
        }
    }
    usage.total = total;
    state.memoryUsage = usage;
    
    return usage;
}

function updatePerformancePanel() {
    const timingsDiv = document.getElementById('computeTimings');
    const memoryDiv = document.getElementById('memoryUsage');
    
    if (timingsDiv) {
        let html = '<div class="perf-title">Compute Shader 耗时</div>';
        if (state.supportsTimestampQuery) {
            let total = 0;
            for (const name of computePassNames) {
                const time = state.computePassTimings[name] || 0;
                total += time;
                if (time > 0.001) {
                    html += `<div class="perf-row"><span>${name}</span><span>${time.toFixed(2)} ms</span></div>`;
                }
            }
            html += `<div class="perf-row total"><span>Total</span><span>${total.toFixed(2)} ms</span></div>`;
        } else {
            html += '<div class="perf-row" style="opacity: 0.6;"><span>不支持 timestamp-query</span></div>';
            html += '<div class="perf-row" style="opacity: 0.6; font-size: 10px;"><span>请启用 WebGPU 时间戳查询功能</span></div>';
        }
        timingsDiv.innerHTML = html;
    }
    
    if (memoryDiv) {
        const usage = calculateMemoryUsage();
        let html = '<div class="perf-title">GPU 内存占用</div>';
        for (const [name, size] of Object.entries(usage)) {
            if (name === 'total') continue;
            const kb = size / 1024;
            html += `<div class="perf-row"><span>${name}</span><span>${kb.toFixed(1)} KB</span></div>`;
        }
        const totalKb = usage.total / 1024;
        html += `<div class="perf-row total"><span>Total</span><span>${totalKb.toFixed(1)} KB</span></div>`;
        memoryDiv.innerHTML = html;
    }
}

function frame() {
    try {
        const device = state.device;
        const context = state.context;
        
        updateParamsBuffer();
        updateRenderUniforms();
        
        const particleWorkgroups = Math.ceil(state.activeParticleCount / 256);
        const gridWorkgroups = Math.ceil(GRID_RESOLUTION * GRID_RESOLUTION / 256);
        const heightfieldWorkgroups = Math.ceil(HEIGHTFIELD_RESOLUTION * HEIGHTFIELD_RESOLUTION / 256);
        
        const encoder = device.createCommandEncoder();
        
        queryIndex = 0;
        
        dispatchComputeWithQuery(encoder, state.computePipelines.updateObstacles, 1, queryIndex++);
        
        dispatchComputeWithQuery(encoder, state.computePipelines.clearGrid, gridWorkgroups, queryIndex++);
        dispatchComputeWithQuery(encoder, state.computePipelines.countGridCells, particleWorkgroups, queryIndex++);
        dispatchComputeWithQuery(encoder, state.computePipelines.buildGridOffsets, 1, queryIndex++);
        dispatchComputeWithQuery(encoder, state.computePipelines.clearGridCounts, gridWorkgroups, queryIndex++);
        dispatchComputeWithQuery(encoder, state.computePipelines.reorderParticles, particleWorkgroups, queryIndex++);
        
        dispatchComputeWithQuery(encoder, state.computePipelines.computeDensityPressure, particleWorkgroups, queryIndex++);
        dispatchComputeWithQuery(encoder, state.computePipelines.computeForces, particleWorkgroups, queryIndex++);
        dispatchComputeWithQuery(encoder, state.computePipelines.integrate, particleWorkgroups, queryIndex++);
        dispatchComputeWithQuery(encoder, state.computePipelines.addParticles, particleWorkgroups, queryIndex++);
        
        dispatchComputeWithQuery(encoder, state.computePipelines.erodeHeightfield, particleWorkgroups, queryIndex++);
        
        encoder.copyBufferToBuffer(
            state.heightfieldBuffer, 0,
            state.heightfieldRenderBuffer, 0,
            HEIGHTFIELD_RESOLUTION * HEIGHTFIELD_RESOLUTION * 4
        );
        
        if (state.supportsTimestampQuery && state.querySet && state.timestampApiAvailable) {
            try {
                encoder.resolveQuerySet(
                    state.querySet, 0, queryIndex * 2,
                    state.queryBuffer, 0
                );
                
                encoder.copyBufferToBuffer(
                    state.queryBuffer, 0,
                    state.queryResolveBuffer, 0,
                    queryIndex * 2 * 8
                );
            } catch (e) {
                console.warn('Query resolve failed, disabling timestamp queries:', e.message);
                state.timestampApiAvailable = false;
            }
        }
        
        const view = context.getCurrentTexture().createView();
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view,
                clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });
        
        renderPass.setPipeline(state.heightfieldRenderPipeline);
        renderPass.setBindGroup(0, state.heightfieldRenderBindGroup);
        renderPass.draw(6, HEIGHTFIELD_RESOLUTION * HEIGHTFIELD_RESOLUTION);
        
        renderPass.setPipeline(state.particleRenderPipeline);
        renderPass.setBindGroup(0, state.particleRenderBindGroup);
        renderPass.draw(6, state.activeParticleCount);
        
        renderPass.setPipeline(state.obstacleRenderPipeline);
        renderPass.setBindGroup(0, state.obstacleRenderBindGroup);
        renderPass.draw(6, MAX_OBSTACLES);
        
        renderPass.end();
        
        device.queue.submit([encoder.finish()]);
        
        if (state.frameCount % 30 === 0) {
            if (state.supportsTimestampQuery) {
                readTimestamps();
            } else {
                updatePerformancePanel();
            }
        }
        
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

function addObstacle(x, y, radius, isRotating = true) {
    if (state.obstacles.length >= MAX_OBSTACLES) return;
    
    for (const obs of state.obstacles) {
        const dx = obs.x - x;
        const dy = obs.y - y;
        if (Math.sqrt(dx * dx + dy * dy) < obs.radius + radius * 0.5) {
            return;
        }
    }
    
    console.log('Adding obstacle at:', x, y, 'radius:', radius);
    state.obstacles.push({ 
        x, y, radius, 
        rotation: 0, 
        angularVelocity: isRotating ? params.rotationSpeed : 0 
    });
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
    
    const couplingFolder = gui.addFolder('流体-固体耦合');
    couplingFolder.add(params, 'rotationSpeed', 0.0, 10.0, 0.1).name('旋转角速度').onChange(() => {
        for (const obs of state.obstacles) {
            if (obs.angularVelocity !== 0) {
                obs.angularVelocity = params.rotationSpeed;
            }
        }
        updateObstaclesBuffer();
    });
    couplingFolder.add(params, 'splashStrength', 0.0, 2.0, 0.01).name('飞溅强度');
    couplingFolder.add(params, 'erosionStrength', 0.0, 1.0, 0.01).name('侵蚀强度');
    couplingFolder.add(params, 'depositionStrength', 0.0, 0.5, 0.01).name('沉积强度');
    
    const renderFolder = gui.addFolder('渲染参数');
    renderFolder.add(params, 'particleRadius', 0.003, 0.02, 0.001).name('粒子半径');
    
    const actions = {
        reset: () => resetSimulation(),
        clearObstacles: () => {
            state.obstacles = [];
            updateObstaclesBuffer();
        },
        addRotatingObstacle: () => {
            addObstacle(0.6, 0.5, 0.08, true);
        },
        addStaticObstacle: () => {
            addObstacle(0.6, 0.5, 0.08, false);
        },
        resetHeightfield: () => {
            const heightfieldData = new Float32Array(HEIGHTFIELD_RESOLUTION * HEIGHTFIELD_RESOLUTION);
            for (let y = 0; y < HEIGHTFIELD_RESOLUTION; y++) {
                for (let x = 0; x < HEIGHTFIELD_RESOLUTION; x++) {
                    const nx = x / HEIGHTFIELD_RESOLUTION;
                    const ny = y / HEIGHTFIELD_RESOLUTION;
                    let height = 0.1 * (1.0 - ny);
                    if (nx > 0.3 && nx < 0.7 && ny > 0.2 && ny < 0.3) {
                        height += 0.1;
                    }
                    heightfieldData[y * HEIGHTFIELD_RESOLUTION + x] = height;
                }
            }
            state.device.queue.writeBuffer(state.heightfieldBuffer, 0, heightfieldData);
        }
    };
    
    gui.add(actions, 'addRotatingObstacle').name('添加旋转障碍物');
    gui.add(actions, 'addStaticObstacle').name('添加静态障碍物');
    gui.add(actions, 'resetHeightfield').name('重置地形');
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
