struct Uniforms {
    resolution: vec2<f32>,
    particleRadius: f32,
    _pad: f32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> heightfield: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) height: f32
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    var out: VertexOutput;
    
    let res: u32 = 128u;
    let x: u32 = instanceIndex % res;
    let y: u32 = instanceIndex / res;
    
    var corner: vec2<f32>;
    switch (vertexIndex) {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    
    let uv: vec2<f32> = (vec2<f32>(f32(x) + corner.x, f32(y) + corner.y)) / vec2<f32>(f32(res));
    let heightIdx: u32 = y * res + x;
    let height: f32 = heightfield[heightIdx];
    
    let aspect: f32 = uniforms.resolution.x / uniforms.resolution.y;
    let cellSize: vec2<f32> = vec2<f32>(1.0 / f32(res), 1.0 / f32(res));
    
    var screenPos: vec2<f32> = uv * 2.0 - 1.0;
    screenPos.x = screenPos.x / aspect;
    
    let heightOffset: f32 = height * 0.5;
    screenPos.y += heightOffset * 0.1;
    
    out.position = vec4<f32>(screenPos, 0.0, 1.0);
    out.uv = uv;
    out.height = height;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var baseColor: vec3<f32> = vec3<f32>(0.3, 0.25, 0.2);
    
    let heightFactor: f32 = clamp(in.height * 5.0, 0.0, 1.0);
    var sandColor: vec3<f32> = vec3<f32>(0.76, 0.7, 0.5);
    var dirtColor: vec3<f32> = vec3<f32>(0.4, 0.3, 0.2);
    
    var color: vec3<f32> = mix(sandColor, dirtColor, heightFactor);
    
    let gridX: f32 = abs(fract(in.uv.x * 128.0) - 0.5);
    let gridY: f32 = abs(fract(in.uv.y * 128.0) - 0.5);
    let gridLine: f32 = smoothstep(0.0, 0.02, min(gridX, gridY));
    
    color = mix(color * 0.7, color, gridLine);
    
    return vec4<f32>(color, 0.8);
}
