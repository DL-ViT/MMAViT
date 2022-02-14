import numpy as np
import torch


def createLDS(input_data, lds_size, STABILIZER=False, HLDS=0, hlds_channels=0):
    # % % HLDS = 0
    # % % lds_size = 4
    # % % Stablilizer = False
    # % % hlds_channels = 0

    if (HLDS == 0):
        block = torch.flip(torch.transpose(input_data, 1, 0), [1]).unsqueeze(0)
        # print(block.shape)
        input_data = block

        _, h, w = input_data.shape
        c0 = block.mean(dim=-1)
        print(c0)

        Y = (input_data - c0).squeeze(0)
        print(Y.shape)
        U, S, V = torch.linalg.svd(Y)
        V = torch.transpose(V, 1, 0)
        # print(f'u {U.shape} S {S.shape} V {V.shape}')
        U = U[:, :lds_size]
        S = S[:lds_size]
        V = V[:, :lds_size]
        # print(U)
        # print(S)
        # print(V)
        CLDS = U

        Z = torch.matmul(torch.diag(S), V.transpose(0, 1))

        A1 = Z[:, 0:Z.shape[-1] - 1]
        A2 = Z[:, 1:Z.shape[-1]]

        X1_t = A1.transpose(0, 1)
        A1inv = torch.linalg.pinv(torch.matmul(A1, X1_t))
        ALDS = torch.matmul(torch.matmul(A2, X1_t), A1inv)

        return ALDS, CLDS


def observability_matrix(A, C, m):
    Om = torch.cat([C, torch.matmul(C, A)], dim=0)
    # print(Om.shape)
    for i in range(1, m - 1, 1):
        CA_product = torch.matmul(C, torch.pow(A, i))
        Om = torch.cat([Om, CA_product], dim=0)
    # #print(Om)
    return Om


def Bobservability_matrix(A, C, m):
    # print(torch.matmul(C, A).shape,C.shape )

    Om = torch.cat([C, torch.matmul(C, A)], dim=-1)
    # print(Om.shape)
    for i in range(1, m - 1, 1):
        CA_product = torch.pow(torch.matmul(C, A), i)
        Om = torch.cat([Om, CA_product], dim=-1)
    # #print(Om)
    return Om


def Bgrassmanian_point(Om):
    Q, R = torch.linalg.qr(Om)

    return Q


def Batch_createLDS(input_data, lds_size, STABILIZER=False, HLDS=0, hlds_channels=0):
    # % % HLDS = 0
    # % % lds_size = 4
    # % % Stablilizer = False
    # % % hlds_channels = 0

    if (HLDS == 0):
        block = torch.flip(torch.transpose(input_data, -1, -2), [-1])
        input_data = block

        bs = block.shape[0]
        T = block.shape[1]
        d = block.shape[2]

        mean_ = torch.mean(block.reshape(bs, T, d, -1), dim=-1).unsqueeze(-1)

        Y = (input_data) - mean_

        U, S, V = torch.linalg.svd(Y)

        V = V.transpose(-2, -1)

        U = U[..., :lds_size]
        S = S[..., :lds_size]
        V = V[..., :lds_size]
        S = torch.diag_embed(S)

        CLDS = U

        Z = torch.matmul(S, V.transpose(-2, -1))

        A1 = Z[..., 0:Z.shape[-1] - 1]
        A2 = Z[..., 1:Z.shape[-1]]

        X1_t = torch.transpose(A1, -1, -2)
        A1inv = torch.linalg.pinv(torch.matmul(A1, X1_t))
        ALDS = torch.matmul(torch.matmul(A2, X1_t), A1inv)

        return ALDS, CLDS


def grassmanian_point(Om):
    Q, R = torch.linalg.qr(Om)
    return Q, R


def image_to_Om(input_tensor, lds_size=3, m=3, num_channels=3):
    assert len(input_tensor.shape) == 3, print(len(input_tensor))
    b, n, c = input_tensor.shape

    OM = torch.FloatTensor()

    for i in range(b):
        Omtensor = []
        for j in range(n):
            lds_input = input_tensor[i, j, :]
            # print(lds_input.shape)
            lds_input = lds_input.view(-1, lds_size)
            ALDS, CLDS = createLDS(lds_input, lds_size, False, 0, 0)
            # print(f'A {ALDS.shape} C {CLDS.shape}' )
            Om = observability_matrix(ALDS, CLDS, m=m)
            # gp,_ = grassmanian_point(Om)
            # print(f"Om {Om.shape} G {gp.shape} ")
            Omtensor.append(Om)
        Omtensor = torch.stack(Omtensor, dim=0).unsqueeze(0)
        # #print(Omtensor.shape)
        # #print(Om.shape,OM.shape)
        OM = torch.cat((OM, Omtensor))
    # #print(Om.shape,Omtensor.shape,OM.shape)
    b, T, h, w = OM.shape
    # print('OM   ',OM.shape)

    return OM.view(b, T, -1)


def batch_image_to_Om(input_tensor, lds_size=3, m=3):
    b, T, c = input_tensor.shape

    input_tensor = input_tensor.view(b, T, -1, lds_size)
    # print(input_tensor.shape)

    ALDS, CLDS = Batch_createLDS(input_tensor, lds_size, False, 0, 0)

    OM = Bobservability_matrix(ALDS, CLDS, m)

    OM = OM.view(b, T, -1)

    return OM


def batch_image_to_Om1(input_tensor, lds_size=3, m=3, num_channels=3):
    assert len(input_tensor.shape) == 3, print(len(input_tensor))
    b, T, c = input_tensor.shape

    lds_input = input_tensor.view(b * T, -1, num_channels)

    ALDS, CLDS = Batch_createLDS(lds_input, lds_size, False, 0, 0)
    # print(ALDS.shape, CLDS.shape)
    OM = Bobservability_matrix(ALDS, CLDS, m)
    OM = OM.view(b, T, -1)
    # Q, R = grassmanian_point(OM)
    # print(OM.shape)

    # print(Q.shape)
    # b_T, c1, c2 = Q.shape
    # Q = Q.view(b,T,c1*c2)
    # print(OM.shape,Q.shape)

    return OM


def test():
    torch.manual_seed(0)
    np.random.seed(0)

    om = batch_image_to_Om(torch.randn((4, 64, 12)), lds_size=3, m=3)
    print(om.shape)
    exit()

    lds_input = np.arange(16).reshape(4, 4)  # .transpose()

    lds_input = torch.from_numpy(np.array([0.730330862855453, 0.458848828179931, 0.231594386708524, 0.395515215668593,
                                           0.488608973803579, 0.963088539286913, 0.488897743920167, 0.367436648544477,
                                           0.578525061023439, 0.546805718738968, 0.624060088173690, 0.987982003161633,
                                           0.237283579771521, 0.521135830804002, 0.679135540865748,
                                           0.0377388662395521]).reshape(4, 4))
    # print(lds_input.shape)

    ALDS, CLDS = createLDS(lds_input, lds_size=3)
    print(ALDS, CLDS)
    # print(OM.shape)
    # patch_shape = 16
    # unfold = torch.nn.Unfold(kernel_size=patch_shape,stride = patch_shape)
    # output = unfold(lds_input)
    # #print(output.size())
    # Omtensor = []
    # for i in range(output.shape[-1]):
    #     lds_input = output[:,:,i]
    #     #print(lds_input.shape)
    #     lds_input = lds_input.view(-1,16)
    #     ALDS, CLDS = createLDS (lds_input, 3, False, 0, 0)
    #     #print(ALDS.shape,CLDS.shape)
    #     Om = observability_matrix(ALDS, CLDS, 3)
    #     gp = grassmanian_point(Om)
    #     #print(Om.shape,gp.shape)
    #     Omtensor.append(Om)
    # Omtensor = torch.stack(Omtensor,dim=0)
    # #print(Omtensor.shape)
    # input_data = torch.from_numpy(np.arange(16).reshape(4, 4)).float()
    # input_data = torch.randn((16 * 16, 4))  # .unsqueeze(0)
    # input_data = torch.stack((input_data,input_data),dim=0)
    # print(input_data)
    # # if len(input_data.shape) == 3:
    # #     i = input_data.permute(0, 2, 1)
    # # else:
    # #     i = input_data.permute(1, 0)
    # i = torch.transpose(input_data,-1,-2)
    # #print(i)
    #
    # # block = torch.flip(torch.transpose(input_data,-1,-2), [-1])  # .view(4,4,-1)
    # # #print(block)
    # lds_size = 3
    # # ALDS, CLDS = Batch_createLDS(input_data, lds_size, False, 0, 0)
    # # Om = Bobservability_matrix(ALDS, CLDS, lds_size)
    # A = torch.randn((3, 3))
    # C = torch.randn((3, 4))
    #
    # ALDS, CLDS = createLDS(input_data, 3)
    # # print(ALDS,'\n',CLDS)
    # # print(ALDS.shape,CLDS.shape)
    # Om = observability_matrix(ALDS, CLDS, lds_size)
    # # print(Om)
    # ALDS, CLDS = Batch_createLDS(torch.stack((input_data, input_data), dim=0), lds_size, False, 0, 0)
    # Om = Bobservability_matrix(ALDS, CLDS, lds_size)
    # # print(Om)
    # # print(ALDS,'\n',CLDS)
    # # print(ALDS.shape,CLDS.shape)

# cpkt = torch.load('/home/iliask/PycharmProjects/Compact-Transformers/output/train/20211110-132218
# -grassmanian_vit_2_4_32-32/model_best.pth.tar')
# print(cpkt.keys())
# print(cpkt['arch'])
# print(cpkt['state_dict'].keys())
# test()
# import matplotlib.pyplot as plt
# # helpers
#
#
# import seaborn
# def draw(data:torch.Tensor):
#     seaborn.heatmap(data.detach().cpu().numpy())
#     plt.show()
#
# draw(torch.randn(10,10))
