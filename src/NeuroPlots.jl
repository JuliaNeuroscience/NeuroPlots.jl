module NeuroPlots


using Makie
import Statistics: mean
import GMT: triangulate

export plot_topography

"""
    plot_topography(channelList::Vector{String}, channelValues::Vector{Number}; gridSize = 1000)

Plots a topographical map of the head over the desired points 
given by `channelList` and their assigned `channelValues`, 
with default grid size being `gridSize = 1000`.

# Example
```jldoctest
channelExamples = String["Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2"];

plot_topography(channelExamples, rand(length(channelExamples)))
```
"""
function plot_topography(channelList::Vector{String}, channelValues::Vector{T}; gridSize = 1000) where T <: Number

    # Basic checking
    length(channelList) == length(channelValues) || error(
        "The lengths of channelList($(length(channelList))) and 
         channelValues($(length(channelValues))) should be the SAME.",
    )

    # Retrieve current `channelList` information from the standard channel information
    standardChannelLabel, standardCoordinateX, standardCoordinateY = standardChannels.label, standardChannels.x, standardChannels.y
    channelIndex = indexin(channelList, standardChannelLabel)
    channelXs, channelYs =
        standardCoordinateX[channelIndex], standardCoordinateY[channelIndex]
    maxX, maxY = maximum(channelXs), maximum(channelYs)
    channelPositions = Point2f.(channelXs, channelYs)

    # Extend data beyond the boundary, see: 
    # https://www.mathworks.com/matlabcentral/fileexchange/72729-topographic-eeg-meg-plot
    extendData = append_nearest_values(channelXs, channelYs, channelValues)

    # Interpolate data with function `triangualte` from `GMT.jl``, 
    # see: `gmthelp(triangulate)` or `?triangulate` for more information.
    interpolateData =
        triangulate(extendData, region = "-1/1/-1/1", inc = string(gridSize, "+n"))
    xs, ys, zs = interpolateData.x, interpolateData.y, interpolateData.z

    # Remove data outside of circle
    zs[abs.(complex.(repeat(xs, outer = gridSize), repeat(ys, inner = gridSize))) .> 1] .= NaN

    # Initianize the figure
    fig = Figure(resolution = (1300, 1200))
    ga = fig[1, 1] = GridLayout()
    gb = fig[1, 2] = GridLayout()

    ax1 = Axis(
        ga[1, 1],
        xzoomlock = true,
        yzoomlock = true,
        xgridvisible = false,
        ygridvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        bottomspinevisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
    )

    # Draw the contour
    contourf!(ax1, xs, ys, zs)

    # Draw the scatter of the electrodes
    scatter!(
        ax1,
        channelPositions,
        color = :black,
        strokecolor = :black,
        strokewidth = 2,
        markersize = 6,
    )

    # Draw the labels of the electrodes
    text!(ax1, channelList, position = channelPositions)

    # Draw the circle around the head
    poly!(
        ax1,
        Circle(Point2f(0, 0), maxX),
        color = :transparent,
        strokecolor = :black,
        strokewidth = 4,
    )

    # Draw the nose
    lines!(ax1, nosePoints(-10, maxX), color = :black, linewidth = 4)
    lines!(ax1, nosePoints(+10, maxX), color = :black, linewidth = 4)

    # Draw the ears
    lines!(ax1, earPoints(-0.75, maxX), color = :black, linewidth = 4)
    lines!(ax1, earPoints(+0.75, maxX), color = :black, linewidth = 4)

    # Add Colorbar
    Colorbar(gb[1, 1], width = 20, ticks = 0:0.1:1)

    # Some decorations
    colsize!(fig.layout, 1, Aspect(1, 1.0))

    save("figure.png", fig, px_per_unit = 1) # output size = 1300 x 1000 pixels

    return fig
end

"""
    append_nearest_values(Xs, Ys, Vals; n = 8, radius = 1.2, k = 4)

Append `n = 8` extra points to the original data at a circle with `radius = 1.2`. 
The values of these points are calculated from the nearst `k = 4` points' values
`Vals` at the original coordinates `(Xs, Ys)`. 
"""
function append_nearest_values(Xs, Ys, Vals; n = 8, radius = 1.2, k = 4)
    newPoints = Array{Float64,2}(undef, n, 3)
    for i = 1:n
        coordinates = radius * exp(i * 2pi * im / n) # Euler's identity
        distances = coordinates .- complex.(Xs, Ys)
        newVal = mean(Vals[sortperm(abs.(distances)).<= k])
        newPoints[i, :] = [real(coordinates), imag(coordinates), newVal]
    end
    return vcat([Xs Ys Vals], newPoints)
end

"""
    nosePoints(angle, radius)

Generate two points to define the left (`angle > 0`) or right (`angle < 0`) line 
to draw the contour of the avator nose, starting from the surface with `radius`.
"""
nosePoints(angle, radius) =
    [radius * Point2f(cosd(90 + angle), sind(90 + angle)), 0.85 * Point2f(0, 1)]

"""
    earPoints(focusX, radius; npoints = 100)

Generate `npoints = 100` points to define the ellipse curve of the left (`FocusX < 0`) 
or right (``FocusX > 0`) ear, with the focus point being at `(FocusX, 0)`.
"""
function earPoints(focusX, radius; npoints = 100)
    coordinates = exp.(collect(1:npoints+1) * 2pi * im / npoints) # Euler's identity
    modifyCoordinates =
        complex.(focusX .+ 0.09 .* real(coordinates), 0.18 .* imag(coordinates))
    modifyCoordinates[abs.(modifyCoordinates).< radius] .= NaN
    return Point2f.(real(modifyCoordinates), imag(modifyCoordinates))
end

"""
    standardChannels

Standard 1005 EEG electrode positions. from: https://github.com/sappelhoff/eeg_positions/blob/main/data/Nz-T10-Iz-T9/standard_1005_2D.tsv
"""
const standardChannels = (label = String["AF1", "AF10", "AF10h", "AF1h", "AF2", "AF2h", "AF3", "AF3h", "AF4", "AF4h", "AF5", "AF5h", "AF6", "AF6h", "AF7", "AF7h", "AF8", "AF8h", "AF9", "AF9h", "AFF1", "AFF10", "AFF10h", "AFF1h", "AFF2", "AFF2h", "AFF3", "AFF3h", "AFF4", "AFF4h", "AFF5", "AFF5h", "AFF6", "AFF6h", "AFF7", "AFF7h", "AFF8", "AFF8h", "AFF9", "AFF9h", "AFFz", "AFp1", "AFp10", "AFp10h", "AFp1h", "AFp2", "AFp2h", "AFp3", "AFp3h", "AFp4", "AFp4h", "AFp5", "AFp5h", "AFp6", "AFp6h", "AFp7", "AFp7h", "AFp8", "AFp8h", "AFp9", "AFp9h", "AFpz", "AFz", "C1", "C1h", "C2", "C2h", "C3", "C3h", "C4", "C4h", "C5", "C5h", "C6", "C6h", "CCP1", "CCP1h", "CCP2", "CCP2h", "CCP3", "CCP3h", "CCP4", "CCP4h", "CCP5", "CCP5h", "CCP6", "CCP6h", "CCPz", "CP1", "CP1h", "CP2", "CP2h", "CP3", "CP3h", "CP4", "CP4h", "CP5", "CP5h", "CP6", "CP6h", "CPP1", "CPP1h", "CPP2", "CPP2h", "CPP3", "CPP3h", "CPP4", "CPP4h", "CPP5", "CPP5h", "CPP6", "CPP6h", "CPPz", "CPz", "Cz", "F1", "F10", "F10h", "F1h", "F2", "F2h", "F3", "F3h", "F4", "F4h", "F5", "F5h", "F6", "F6h", "F7", "F7h", "F8", "F8h", "F9", "F9h", "FC1", "FC1h", "FC2", "FC2h", "FC3", "FC3h", "FC4", "FC4h", "FC5", "FC5h", "FC6", "FC6h", "FCC1", "FCC1h", "FCC2", "FCC2h", "FCC3", "FCC3h", "FCC4", "FCC4h", "FCC5", "FCC5h", "FCC6", "FCC6h", "FCCz", "FCz", "FFC1", "FFC1h", "FFC2", "FFC2h", "FFC3", "FFC3h", "FFC4", "FFC4h", "FFC5", "FFC5h", "FFC6", "FFC6h", "FFCz", "FFT10", "FFT10h", "FFT7", "FFT7h", "FFT8", "FFT8h", "FFT9", "FFT9h", "FT10", "FT10h", "FT7", "FT7h", "FT8", "FT8h", "FT9", "FT9h", "FTT10", "FTT10h", "FTT7", "FTT7h", "FTT8", "FTT8h", "FTT9", "FTT9h", "Fp1", "Fp1h", "Fp2", "Fp2h", "Fpz", "Fz", "I1", "I1h", "I2", "I2h", "Iz", "LPA", "N1", "N1h", "N2", "N2h", "NAS", "NFp1", "NFp1h", "NFp2", "NFp2h", "NFpz", "Nz", "O1", "O1h", "O2", "O2h", "OI1", "OI1h", "OI2", "OI2h", "OIz", "Oz", "P1", "P10", "P10h", "P1h", "P2", "P2h", "P3", "P3h", "P4", "P4h", "P5", "P5h", "P6", "P6h", "P7", "P7h", "P8", "P8h", "P9", "P9h", "PO1", "PO10", "PO10h", "PO1h", "PO2", "PO2h", "PO3", "PO3h", "PO4", "PO4h", "PO5", "PO5h", "PO6", "PO6h", "PO7", "PO7h", "PO8", "PO8h", "PO9", "PO9h", "POO1", "POO10", "POO10h", "POO1h", "POO2", "POO2h", "POO3", "POO3h", "POO4", "POO4h", "POO5", "POO5h", "POO6", "POO6h", "POO7", "POO7h", "POO8", "POO8h", "POO9", "POO9h", "POOz", "POz", "PPO1", "PPO10", "PPO10h", "PPO1h", "PPO2", "PPO2h", "PPO3", "PPO3h", "PPO4", "PPO4h", "PPO5", "PPO5h", "PPO6", "PPO6h", "PPO7", "PPO7h", "PPO8", "PPO8h", "PPO9", "PPO9h", "PPOz", "Pz", "RPA", "T10", "T10h", "T7", "T7h", "T8", "T8h", "T9", "T9h", "TP10", "TP10h", "TP7", "TP7h", "TP8", "TP8h", "TP9", "TP9h", "TPP10", "TPP10h", "TPP7", "TPP7h", "TPP8", "TPP8h", "TPP9", "TPP9h", "TTP10", "TTP10h", "TTP7", "TTP7h", "TTP8", "TTP8h", "TTP9", "TTP9h"], x = [-0.1025, 0.5878, 0.5021, -0.0512, 0.1025, 0.0512, -0.2067, -0.1543, 0.2067, 0.1543, -0.3143, -0.26, 0.3143, 0.26, -0.427, -0.3699, 0.427, 0.3699, -0.5878, -0.5021, -0.1207, 0.7071, 0.6039, -0.0602, 0.1207, 0.0602, -0.2444, -0.182, 0.2444, 0.182, -0.3743, -0.3084, 0.3743, 0.3084, -0.5138, -0.4426, 0.5138, 0.4426, -0.7071, -0.6039, 0.0, -0.0803, 0.454, 0.3878, -0.0401, 0.0803, 0.0401, -0.1615, -0.1207, 0.1615, 0.1207, -0.2443, -0.2027, 0.2443, 0.2027, -0.3299, -0.2867, 0.3299, 0.2867, -0.454, -0.3878, 0.0, 0.0, -0.1584, -0.0787, 0.1584, 0.0787, -0.3249, -0.2401, 0.3249, 0.2401, -0.5095, -0.4142, 0.5095, 0.4142, -0.157, -0.078, 0.157, 0.078, -0.3219, -0.2379, 0.3219, 0.2379, -0.5042, -0.4102, 0.5042, 0.4102, 0.0, -0.1527, -0.0759, 0.1527, 0.0759, -0.3126, -0.2313, 0.3126, 0.2313, -0.4881, -0.3978, 0.4881, 0.3978, -0.1455, -0.0724, 0.1455, 0.0724, -0.2969, -0.22, 0.2969, 0.22, -0.4612, -0.377, 0.4612, 0.377, 0.0, 0.0, 0.0, -0.1349, 0.809, 0.691, -0.0672, 0.1349, 0.0672, -0.2744, -0.2038, 0.2744, 0.2038, -0.4234, -0.3474, 0.4234, 0.3474, -0.5879, -0.5032, 0.5879, 0.5032, -0.809, -0.691, -0.1527, -0.0759, 0.1527, 0.0759, -0.3126, -0.2313, 0.3126, 0.2313, -0.4881, -0.3978, 0.4881, 0.3978, -0.157, -0.078, 0.157, 0.078, -0.3219, -0.2379, 0.3219, 0.2379, -0.5042, -0.4102, 0.5042, 0.4102, 0.0, 0.0, -0.1455, -0.0724, 0.1455, 0.0724, -0.2969, -0.22, 0.2969, 0.22, -0.4612, -0.377, 0.4612, 0.377, 0.0, 0.891, 0.761, -0.6474, -0.5509, 0.6474, 0.5509, -0.891, -0.761, 0.9511, 0.8123, -0.691, -0.5852, 0.691, 0.5852, -0.9511, -0.8123, 0.9877, 0.8436, -0.7176, -0.6059, 0.7176, 0.6059, -0.9877, -0.8436, -0.2245, -0.1137, 0.2245, 0.1137, 0.0, 0.0, -0.309, -0.1564, 0.309, 0.1564, 0.0, -1.0, -0.309, -0.1564, 0.309, 0.1564, 0.0, -0.2639, -0.1336, 0.2639, 0.1336, 0.0, 0.0, -0.2245, -0.1137, 0.2245, 0.1137, -0.2639, -0.1336, 0.2639, 0.1336, 0.0, 0.0, -0.1349, 0.809, 0.691, -0.0672, 0.1349, 0.0672, -0.2744, -0.2038, 0.2744, 0.2038, -0.4234, -0.3474, 0.4234, 0.3474, -0.5879, -0.5032, 0.5879, 0.5032, -0.809, -0.691, -0.1025, 0.5878, 0.5021, -0.0512, 0.1025, 0.0512, -0.2067, -0.1543, 0.2067, 0.1543, -0.3143, -0.26, 0.3143, 0.26, -0.427, -0.3699, 0.427, 0.3699, -0.5878, -0.5021, -0.0803, 0.454, 0.3878, -0.0401, 0.0803, 0.0401, -0.1615, -0.1207, 0.1615, 0.1207, -0.2443, -0.2027, 0.2443, 0.2027, -0.3299, -0.2867, 0.3299, 0.2867, -0.454, -0.3878, 0.0, 0.0, -0.1207, 0.7071, 0.6039, -0.0602, 0.1207, 0.0602, -0.2444, -0.182, 0.2444, 0.182, -0.3743, -0.3084, 0.3743, 0.3084, -0.5138, -0.4426, 0.5138, 0.4426, -0.7071, -0.6039, 0.0, 0.0, 1.0, 1.0, 0.8541, -0.7266, -0.6128, 0.7266, 0.6128, -1.0, -0.8541, 0.9511, 0.8123, -0.691, -0.5852, 0.691, 0.5852, -0.9511, -0.8123, 0.891, 0.761, -0.6474, -0.5509, 0.6474, 0.5509, -0.891, -0.761, 0.9877, 0.8436, -0.7176, -0.6059, 0.7176, 0.6059, -0.9877, -0.8436], y = [0.5139, 0.809, 0.691, 0.5105, 0.5139, 0.5105, 0.5274, 0.5194, 0.5274, 0.5194, 0.5513, 0.5379, 0.5513, 0.5379, 0.5879, 0.5678, 0.5879, 0.5678, 0.809, 0.691, 0.4195, 0.7071, 0.6039, 0.4155, 0.4195, 0.4155, 0.4361, 0.4263, 0.4361, 0.4263, 0.4661, 0.4493, 0.4661, 0.4493, 0.5138, 0.4874, 0.5138, 0.4874, 0.7071, 0.6039, 0.4142, 0.6148, 0.891, 0.761, 0.6133, 0.6148, 0.6133, 0.621, 0.6174, 0.621, 0.6174, 0.6317, 0.6258, 0.6317, 0.6258, 0.6474, 0.6388, 0.6474, 0.6388, 0.891, 0.761, 0.6128, 0.5095, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0804, -0.0791, -0.0804, -0.0791, -0.0857, -0.0825, -0.0857, -0.0825, -0.096, -0.0901, -0.096, -0.0901, -0.0787, -0.1616, -0.1591, -0.1616, -0.1591, -0.1718, -0.1657, -0.1718, -0.1657, -0.1912, -0.1802, -0.1912, -0.1802, -0.2445, -0.2412, -0.2445, -0.2412, -0.2587, -0.2503, -0.2587, -0.2503, -0.2852, -0.2702, -0.2852, -0.2702, -0.2401, -0.1584, 0.0, 0.3302, 0.5878, 0.5021, 0.3262, 0.3302, 0.3262, 0.3467, 0.3369, 0.3467, 0.3369, 0.3771, 0.3599, 0.3771, 0.3599, 0.427, 0.3992, 0.427, 0.3992, 0.5878, 0.5021, 0.1616, 0.1591, 0.1616, 0.1591, 0.1718, 0.1657, 0.1718, 0.1657, 0.1912, 0.1802, 0.1912, 0.1802, 0.0804, 0.0791, 0.0804, 0.0791, 0.0857, 0.0825, 0.0857, 0.0825, 0.096, 0.0901, 0.096, 0.0901, 0.0787, 0.1584, 0.2445, 0.2412, 0.2445, 0.2412, 0.2587, 0.2503, 0.2587, 0.2503, 0.2852, 0.2702, 0.2852, 0.2702, 0.2401, 0.454, 0.3878, 0.3299, 0.3048, 0.3299, 0.3048, 0.454, 0.3878, 0.309, 0.2639, 0.2245, 0.2057, 0.2245, 0.2057, 0.309, 0.2639, 0.1564, 0.1336, 0.1137, 0.1036, 0.1137, 0.1036, 0.1564, 0.1336, 0.691, 0.7176, 0.691, 0.7176, 0.7266, 0.3249, -0.9511, -0.9877, -0.9511, -0.9877, -1.0, 0.0, 0.9511, 0.9877, 0.9511, 0.9877, 1.0, 0.8123, 0.8436, 0.8123, 0.8436, 0.8541, 1.0, -0.691, -0.7176, -0.691, -0.7176, -0.8123, -0.8436, -0.8123, -0.8436, -0.8541, -0.7266, -0.3302, -0.5878, -0.5021, -0.3262, -0.3302, -0.3262, -0.3467, -0.3369, -0.3467, -0.3369, -0.3771, -0.3599, -0.3771, -0.3599, -0.427, -0.3992, -0.427, -0.3992, -0.5878, -0.5021, -0.5139, -0.809, -0.691, -0.5105, -0.5139, -0.5105, -0.5274, -0.5194, -0.5274, -0.5194, -0.5513, -0.5379, -0.5513, -0.5379, -0.5879, -0.5678, -0.5879, -0.5678, -0.809, -0.691, -0.6148, -0.891, -0.761, -0.6133, -0.6148, -0.6133, -0.621, -0.6174, -0.621, -0.6174, -0.6317, -0.6258, -0.6317, -0.6258, -0.6474, -0.6388, -0.6474, -0.6388, -0.891, -0.761, -0.6128, -0.5095, -0.4195, -0.7071, -0.6039, -0.4155, -0.4195, -0.4155, -0.4361, -0.4263, -0.4361, -0.4263, -0.4661, -0.4493, -0.4661, -0.4493, -0.5138, -0.4874, -0.5138, -0.4874, -0.7071, -0.6039, -0.4142, -0.3249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.309, -0.2639, -0.2245, -0.2057, -0.2245, -0.2057, -0.309, -0.2639, -0.454, -0.3878, -0.3299, -0.3048, -0.3299, -0.3048, -0.454, -0.3878, -0.1564, -0.1336, -0.1137, -0.1036, -0.1137, -0.1036, -0.1564, -0.1336]);

end # module
