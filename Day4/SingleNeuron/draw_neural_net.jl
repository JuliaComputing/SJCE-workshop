using Plots; gr()

function draw_neuron(x, y, r; c=:blue)

    θs = 0:0.1:2pi
    xs = x .+ r.*cos.(θs)
    ys = y .+ r.*sin.(θs)


    plot!(xs, ys, seriestype=:shape, c=c, alpha=0.5, aspect_ratio=1, leg=false)

end


neuron_coords(x, N, spacing) = range(-(N - 1)/2 * spacing, spacing, N)


function draw_neurons(x, N, spacing, r; c=:blue)

    ys = neuron_coords(x, N, spacing)

    draw_neuron.(x, ys, r; c=c)

end


function draw_layer(x, spacing, N1, N2, r)

    plot!(framestyle=:none, grid=:none)

    first_x = x
    second_x = x + 1

    first = neuron_coords(x + 1, N1, spacing)
    second = neuron_coords(x, N2, spacing)

    draw_neurons(x, N1, 1, r; c=:blue)
    draw_neurons(x+1, N2, 1, r; c=:red)

    for i in 1:N1
        for j in 1:N2

            vec = [second_x - first_x, second[j] - first[i]]
            normalize!(vec)

            start = [first_x, first[i]] + 1.2*r*vec
            finish = [second_x, second[j]] - 1.2*r*vec


            plot!([start[1], finish[1]], [start[2], finish[2]], c=:black, alpha=0.5)
        end
    end

end
