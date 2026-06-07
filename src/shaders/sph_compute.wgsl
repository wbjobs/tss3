struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
    color: vec3<f32>,
    _pad: f32
};

struct Params {
    particleCount: u32,
    gridSize: f32,
    smoothingRadius: f32,
    restDensity: f32,
    gasConstant: f32,
    viscosity: f32,
    dt: f32,
    gravity: f32,
    boundaryMin: vec2<f32>,
    boundaryMax: vec2<f32>,
    mousePos: vec2<f32>,
    mouseActive: u32,
    mouseMode: u32,
    gridResolution: u32,
    maxParticles: u32,
    _pad1: u32
};

struct Obstacle {
    position: vec2<f32>,
    radius: f32
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> gridIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> gridOffsets: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> obstacles: array<Obstacle>;
@group(0) @binding(5) var<storage, read_write> gridCounts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> gridIndicesTemp: array<u32>;

const PI: f32 = 3.14159265359;
const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> sPos: array<vec2<f32>, 256>;
var<workgroup> sVel: array<vec2<f32>, 256>;
var<workgroup> sDensity: array<f32, 256>;
var<workgroup> sPressure: array<f32, 256>;

fn poly6Kernel(rSq: f32, hSq: f32, poly6Factor: f32) -> f32 {
    let h2MinusR2: f32 = hSq - rSq;
    return poly6Factor * h2MinusR2 * h2MinusR2 * h2MinusR2;
}

fn spikyKernelGradient(diff: vec2<f32>, r: f32, h: f32, spikyFactor: f32) -> vec2<f32> {
    let hMinusR: f32 = h - r;
    let gradScale: f32 = spikyFactor * hMinusR * hMinusR / max(r, 0.0001);
    return diff * gradScale;
}

fn viscosityKernelLaplacian(r: f32, h: f32, viscosityFactor: f32) -> f32 {
    return viscosityFactor * (h - r);
}

fn getGridCell(pos: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(
        i32(floor(pos.x / params.gridSize)),
        i32(floor(pos.y / params.gridSize))
    );
}

fn getGridIndex(cell: vec2<i32>) -> u32 {
    let res: i32 = i32(params.gridResolution);
    let cx: i32 = clamp(cell.x, 0, res - 1);
    let cy: i32 = clamp(cell.y, 0, res - 1);
    return u32(cy * res + cx);
}

@compute @workgroup_size(256)
fn clearGrid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let gridSize: u32 = params.gridResolution * params.gridResolution;
    if (i >= gridSize) { return; }
    gridOffsets[i] = -1;
    atomicStore(&gridCounts[i], 0u);
}

@compute @workgroup_size(256)
fn countGridCells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= params.particleCount) { return; }

    let p: Particle = particles[i];
    let cell: vec2<i32> = getGridCell(p.position);
    let gridIdx: u32 = getGridIndex(cell);
    
    atomicAdd(&gridCounts[gridIdx], 1u);
}

@compute @workgroup_size(1)
fn buildGridOffsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gridSize: u32 = params.gridResolution * params.gridResolution;
    
    var prefixSum: i32 = 0;
    for (var i: u32 = 0u; i < gridSize; i++) {
        let count: u32 = atomicLoad(&gridCounts[i]);
        if (count > 0u) {
            gridOffsets[i] = prefixSum;
            prefixSum += i32(count);
        } else {
            gridOffsets[i] = -1;
        }
    }
}

@compute @workgroup_size(256)
fn reorderParticles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= params.particleCount) { return; }

    let p: Particle = particles[i];
    let cell: vec2<i32> = getGridCell(p.position);
    let gridIdx: u32 = getGridIndex(cell);
    
    let offset: u32 = u32(atomicAdd(&gridCounts[gridIdx], 1u)) + u32(gridOffsets[gridIdx]);
    
    gridIndices[offset * 2] = gridIdx;
    gridIndices[offset * 2 + 1] = i;
}

@compute @workgroup_size(256)
fn clearGridCounts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let gridSize: u32 = params.gridResolution * params.gridResolution;
    if (i >= gridSize) { return; }
    atomicStore(&gridCounts[i], 0u);
}

fn getGridStart(gridIdx: u32) -> u32 {
    let offset: i32 = gridOffsets[gridIdx];
    if (offset < 0) { return params.particleCount; }
    return u32(offset);
}

fn getGridEnd(gridIdx: u32) -> u32 {
    let count: u32 = atomicLoad(&gridCounts[gridIdx]);
    let offset: i32 = gridOffsets[gridIdx];
    if (offset < 0) { return params.particleCount; }
    return u32(offset) + count;
}

fn checkObstacleBoundary(pos: vec2<f32>) -> vec2<f32> {
    var boundaryForce: vec2<f32> = vec2<f32>(0.0);
    
    for (var o: u32 = 0u; o < 64u; o = o + 1u) {
        let obs: Obstacle = obstacles[o];
        if (obs.radius <= 0.0) { break; }
        
        let diff: vec2<f32> = pos - obs.position;
        let distSq: f32 = dot(diff, diff);
        let influenceRadius: f32 = obs.radius + params.smoothingRadius;
        let influenceRadiusSq: f32 = influenceRadius * influenceRadius;
        
        if (distSq < influenceRadiusSq && distSq > 0.000001) {
            let dist: f32 = sqrt(distSq);
            let penetration: f32 = influenceRadius - dist;
            let normal: vec2<f32> = diff / dist;
            boundaryForce += normal * penetration * penetration * 1000.0;
        }
    }
    
    return boundaryForce;
}

@compute @workgroup_size(256)
fn computeDensityPressure(@builtin(global_invocation_id) gid: vec3<u32>,
                          @builtin(local_invocation_id) localId: vec3<u32>,
                          @builtin(workgroup_id) groupId: vec3<u32>) {
    let localIdx: u32 = localId.x;
    let groupStart: u32 = groupId.x * WORKGROUP_SIZE;
    let groupEnd: u32 = min(groupStart + WORKGROUP_SIZE, params.particleCount);
    
    for (var s: u32 = localIdx; s < WORKGROUP_SIZE; s += WORKGROUP_SIZE) {
        let globalIdx: u32 = groupStart + s;
        if (globalIdx < groupEnd) {
            let p: Particle = particles[globalIdx];
            sPos[s] = p.position;
            sVel[s] = p.velocity;
            sDensity[s] = p.density;
            sPressure[s] = p.pressure;
        } else {
            sPos[s] = vec2<f32>(-1000.0, -1000.0);
        }
    }
    workgroupBarrier();
    
    let i: u32 = gid.x;
    if (i >= params.particleCount) { return; }

    let h: f32 = params.smoothingRadius;
    let hSq: f32 = h * h;
    let h9: f32 = h * h * h * h * h * h * h * h * h;
    let poly6Factor: f32 = 315.0 / (64.0 * PI * h9);
    
    let pi: Particle = particles[i];
    let pos: vec2<f32> = pi.position;
    let cell: vec2<i32> = getGridCell(pos);
    
    var density: f32 = 0.0;
    
    for (var s: u32 = 0u; s < WORKGROUP_SIZE; s++) {
        let diff: vec2<f32> = pos - sPos[s];
        let rSq: f32 = dot(diff, diff);
        if (rSq < hSq) {
            density += poly6Kernel(rSq, hSq, poly6Factor);
        }
    }
    
    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let neighborCell: vec2<i32> = cell + vec2<i32>(dx, dy);
            let gridIdx: u32 = getGridIndex(neighborCell);
            
            let start: u32 = getGridStart(gridIdx);
            let end: u32 = getGridEnd(gridIdx);
            
            for (var k: u32 = start; k < end; k++) {
                let j: u32 = gridIndices[k * 2 + 1];
                if (j >= groupStart && j < groupEnd) { continue; }
                
                let pj: Particle = particles[j];
                let diff: vec2<f32> = pos - pj.position;
                let rSq: f32 = dot(diff, diff);
                
                if (rSq < hSq) {
                    density += poly6Kernel(rSq, hSq, poly6Factor);
                }
            }
        }
    }
    
    let selfDensity: f32 = poly6Factor * hSq * hSq * hSq;
    density += selfDensity;
    
    var newPi: Particle = pi;
    newPi.density = density;
    newPi.pressure = params.gasConstant * (density - params.restDensity);
    particles[i] = newPi;
}

@compute @workgroup_size(256)
fn computeForces(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(local_invocation_id) localId: vec3<u32>,
                 @builtin(workgroup_id) groupId: vec3<u32>) {
    let localIdx: u32 = localId.x;
    let groupStart: u32 = groupId.x * WORKGROUP_SIZE;
    let groupEnd: u32 = min(groupStart + WORKGROUP_SIZE, params.particleCount);
    
    for (var s: u32 = localIdx; s < WORKGROUP_SIZE; s += WORKGROUP_SIZE) {
        let globalIdx: u32 = groupStart + s;
        if (globalIdx < groupEnd) {
            let p: Particle = particles[globalIdx];
            sPos[s] = p.position;
            sVel[s] = p.velocity;
            sDensity[s] = p.density;
            sPressure[s] = p.pressure;
        } else {
            sPos[s] = vec2<f32>(-1000.0, -1000.0);
        }
    }
    workgroupBarrier();
    
    let i: u32 = gid.x;
    if (i >= params.particleCount) { return; }

    let h: f32 = params.smoothingRadius;
    let hSq: f32 = h * h;
    let h6: f32 = h * h * h * h * h * h;
    let spikyFactor: f32 = -45.0 / (PI * h6);
    let viscosityFactor: f32 = 45.0 / (PI * h6);
    
    let pi: Particle = particles[i];
    let pos: vec2<f32> = pi.position;
    let myIndexInGroup: u32 = i - groupStart;
    let cell: vec2<i32> = getGridCell(pos);
    
    var pressureForce: vec2<f32> = vec2<f32>(0.0, 0.0);
    var viscosityForce: vec2<f32> = vec2<f32>(0.0, 0.0);
    
    for (var s: u32 = 0u; s < WORKGROUP_SIZE; s++) {
        if (s == myIndexInGroup) { continue; }
        
        let diff: vec2<f32> = pos - sPos[s];
        let rSq: f32 = dot(diff, diff);
        
        if (rSq < hSq && rSq > 0.000001) {
            let r: f32 = sqrt(rSq);
            
            let grad: vec2<f32> = spikyKernelGradient(diff, r, h, spikyFactor);
            let pressureTerm: f32 = (pi.pressure + sPressure[s]) / (2.0 * max(sDensity[s], 0.001));
            pressureForce += pressureTerm * grad;
            
            let laplacian: f32 = viscosityKernelLaplacian(r, h, viscosityFactor);
            viscosityForce += params.viscosity * (sVel[s] - pi.velocity) / max(sDensity[s], 0.001) * laplacian;
        }
    }
    
    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let neighborCell: vec2<i32> = cell + vec2<i32>(dx, dy);
            let gridIdx: u32 = getGridIndex(neighborCell);
            
            let start: u32 = getGridStart(gridIdx);
            let end: u32 = getGridEnd(gridIdx);
            
            for (var k: u32 = start; k < end; k++) {
                let j: u32 = gridIndices[k * 2 + 1];
                if (j >= groupStart && j < groupEnd) { continue; }
                
                let pj: Particle = particles[j];
                let diff: vec2<f32> = pos - pj.position;
                let rSq: f32 = dot(diff, diff);
                
                if (rSq < hSq && rSq > 0.000001) {
                    let r: f32 = sqrt(rSq);
                    
                    let grad: vec2<f32> = spikyKernelGradient(diff, r, h, spikyFactor);
                    let pressureTerm: f32 = (pi.pressure + pj.pressure) / (2.0 * max(pj.density, 0.001));
                    pressureForce += pressureTerm * grad;
                    
                    let laplacian: f32 = viscosityKernelLaplacian(r, h, viscosityFactor);
                    viscosityForce += params.viscosity * (pj.velocity - pi.velocity) / max(pj.density, 0.001) * laplacian;
                }
            }
        }
    }
    
    let obstacleForce: vec2<f32> = checkObstacleBoundary(pos);
    
    var gravity: vec2<f32> = vec2<f32>(0.0, params.gravity);
    var acceleration: vec2<f32> = pressureForce + viscosityForce + gravity + obstacleForce;
    
    if (pi.density > 0.001) {
        acceleration = acceleration / pi.density;
    }
    
    var newPi: Particle = pi;
    newPi.velocity = newPi.velocity + acceleration * params.dt;
    
    let speedSq: f32 = dot(newPi.velocity, newPi.velocity);
    let speed: f32 = sqrt(speedSq);
    let t: f32 = clamp(speed / 5.0, 0.0, 1.0);
    newPi.color = mix(vec3<f32>(0.2, 0.5, 1.0), vec3<f32>(1.0, 0.3, 0.2), t);
    
    particles[i] = newPi;
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= params.particleCount) { return; }

    var p: Particle = particles[i];
    
    p.position = p.position + p.velocity * params.dt;
    
    let damping: f32 = 0.5;
    let margin: f32 = 0.01;
    
    if (p.position.x < params.boundaryMin.x + margin) {
        p.position.x = params.boundaryMin.x + margin;
        p.velocity.x = -p.velocity.x * damping;
    }
    if (p.position.x > params.boundaryMax.x - margin) {
        p.position.x = params.boundaryMax.x - margin;
        p.velocity.x = -p.velocity.x * damping;
    }
    if (p.position.y < params.boundaryMin.y + margin) {
        p.position.y = params.boundaryMin.y + margin;
        p.velocity.y = -p.velocity.y * damping;
    }
    if (p.position.y > params.boundaryMax.y - margin) {
        p.position.y = params.boundaryMax.y - margin;
        p.velocity.y = -p.velocity.y * damping;
    }
    
    for (var o: u32 = 0u; o < 64u; o = o + 1u) {
        let obs: Obstacle = obstacles[o];
        if (obs.radius <= 0.0) { break; }
        
        let diff: vec2<f32> = p.position - obs.position;
        let distSq: f32 = dot(diff, diff);
        let minDist: f32 = obs.radius + 0.02;
        let minDistSq: f32 = minDist * minDist;
        
        if (distSq < minDistSq) {
            let dist: f32 = sqrt(max(distSq, 0.000001));
            let dir: vec2<f32> = diff / dist;
            p.position = obs.position + dir * minDist;
            
            let velAlongNormal: f32 = dot(p.velocity, dir);
            if (velAlongNormal < 0.0) {
                p.velocity = p.velocity - 2.0 * velAlongNormal * dir * 0.8;
            }
            
            let tangent: vec2<f32> = vec2<f32>(-dir.y, dir.x);
            let velAlongTangent: f32 = dot(p.velocity, tangent);
            p.velocity = p.velocity - tangent * velAlongTangent * 0.3;
        }
    }
    
    if (params.mouseActive == 1u && params.mouseMode == 0u) {
        let diff: vec2<f32> = params.mousePos - p.position;
        let distSq: f32 = dot(diff, diff);
        let influenceSq: f32 = 0.15 * 0.15;
        if (distSq < influenceSq) {
            let dist: f32 = sqrt(max(distSq, 0.000001));
            let force: f32 = (0.15 - dist) * 20.0;
            p.velocity = p.velocity + (diff / dist) * force * params.dt;
        }
    }
    
    particles[i] = p;
}

@compute @workgroup_size(256)
fn addParticles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= params.particleCount) { return; }
    
    if (params.mouseActive == 1u && params.mouseMode == 0u && params.particleCount < params.maxParticles) {
        var p: Particle = particles[i];
        if (p.density < 0.0) {
            let angle: f32 = f32(i) * 0.61803398875 * PI * 2.0;
            let radius: f32 = 0.02 * sqrt(f32(i % 100u));
            p.position = params.mousePos + vec2<f32>(cos(angle) * radius, sin(angle) * radius);
            p.velocity = vec2<f32>(0.0, 0.0);
            p.density = 0.0;
            p.pressure = 0.0;
            p.color = vec3<f32>(0.2, 0.5, 1.0);
            particles[i] = p;
        }
    }
}
